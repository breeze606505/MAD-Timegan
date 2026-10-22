"""TimeGAN training and generation with TransGRU multi-generator and pruning."""

import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from .utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Discriminator, Supervisor, MultiGeneratorEnsemble


class BaseModel:
    """Base model for TimeGAN variants."""

    def __init__(self, opt, ori_data, ori_labels=None):
        self.seed(opt.manualseed)
        self.opt = opt
        self.real_data = self._to_numpy_batch(ori_data)
        self.real_labels = None if ori_labels is None else np.asarray(ori_labels, dtype=np.int64).reshape(-1)
        if self.real_labels is not None and len(self.real_labels) != len(self.real_data):
            raise ValueError("Length mismatch between ori_data and ori_labels.")
        # Keep data feature dimension explicit (separate from z/noise dimension).
        if len(self.real_data) > 0:
            self.opt.feature_dim = int(self.real_data.shape[-1])
        else:
            self.opt.feature_dim = int(getattr(self.opt, "z_dim", 1))
        if self.real_labels is not None and len(self.real_labels) > 0:
            unique_labels = sorted(np.unique(self.real_labels).tolist())
            mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.real_labels = np.asarray([mapping[int(label)] for label in self.real_labels], dtype=np.int64)
            self.opt.num_classes = max(int(getattr(self.opt, "num_classes", 0)), len(unique_labels))
            label_counts = np.bincount(self.real_labels, minlength=self.opt.num_classes).astype(np.float64)
            self.label_probs = label_counts / max(1.0, label_counts.sum())
        else:
            self.opt.num_classes = max(0, int(getattr(self.opt, "num_classes", 0)))
            self.label_probs = None
        self.use_class_conditioning = bool(
            getattr(self.opt, "use_class_conditioning", True)
            and self.real_labels is not None
            and self.opt.num_classes > 1
        )
        if not self.use_class_conditioning:
            self.opt.num_classes = 0
        self.ori_data, self.min_val, self.max_val = NormMinMax(self.real_data)
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)
        self.data_num = int(self.real_data.shape[0])
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, "test")
        self.device = torch.device("cuda:0" if self.opt.device == "cuda" else "cpu")
        self._joint_step = 0
        self._joint_total_steps = int(getattr(self.opt, "stage3_iters", getattr(self.opt, "iteration", 1)))
        self.real_class_distance = self._np_class_center_distance(self.real_data, self.real_labels)
        self.seasonal_periods = self._resolve_seasonal_periods()
        if self.seasonal_periods:
            print(f"[Seasonality] periods={self.seasonal_periods}")

    def seed(self, seed_value):
        if seed_value == -1:
            return
        import random

        random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def save_weights(self, epoch):
        weight_dir = os.path.join(self.opt.outf, self.opt.name, "train", "weights")
        os.makedirs(weight_dir, exist_ok=True)

        torch.save({"epoch": epoch + 1, "state_dict": self.nete.state_dict()}, f"{weight_dir}/netE.pth")
        torch.save({"epoch": epoch + 1, "state_dict": self.netr.state_dict()}, f"{weight_dir}/netR.pth")
        torch.save({"epoch": epoch + 1, "state_dict": self.netg.state_dict()}, f"{weight_dir}/netG.pth")
        torch.save({"epoch": epoch + 1, "state_dict": self.netd.state_dict()}, f"{weight_dir}/netD.pth")
        torch.save({"epoch": epoch + 1, "state_dict": self.nets.state_dict()}, f"{weight_dir}/netS.pth")

    def _load_weights(self):
        if self.opt.resume == "":
            return
        resume = self.opt.resume
        self.opt.iter = torch.load(os.path.join(resume, "netG.pth"), map_location=self.device)["epoch"]
        self.nete.load_state_dict(torch.load(os.path.join(resume, "netE.pth"), map_location=self.device)["state_dict"])
        self.netr.load_state_dict(torch.load(os.path.join(resume, "netR.pth"), map_location=self.device)["state_dict"])
        self.netg.load_state_dict(torch.load(os.path.join(resume, "netG.pth"), map_location=self.device)["state_dict"])
        self.netd.load_state_dict(torch.load(os.path.join(resume, "netD.pth"), map_location=self.device)["state_dict"])
        self.nets.load_state_dict(torch.load(os.path.join(resume, "netS.pth"), map_location=self.device)["state_dict"])

    @staticmethod
    def _to_numpy_batch(data, n=None):
        arr = np.asarray(data)
        if arr.dtype == object:
            arr = np.asarray([np.asarray(x) for x in data], dtype=np.float32)
        arr = arr.astype(np.float32)
        if n is not None:
            arr = arr[: int(n)]
        return arr

    @staticmethod
    def _mae_rmse_np(a, b):
        diff = a - b
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        return mae, rmse

    @staticmethod
    def _safe_float(value, default=np.nan):
        if value is None:
            return float(default)
        if isinstance(value, (float, int, np.floating, np.integer)):
            value = float(value)
        elif torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            value = float(value.detach().mean().cpu())
        else:
            value = float(value)
        if not np.isfinite(value):
            return float(default)
        return float(value)

    @staticmethod
    def _np_class_center_distance(data, labels):
        if labels is None:
            return 0.0
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        centers = []
        for label in sorted(np.unique(labels).tolist()):
            mask = labels == label
            if int(np.sum(mask)) < 2:
                continue
            centers.append(data[mask].mean(axis=0).reshape(-1))
        if len(centers) <= 1:
            return 0.0
        centers = np.asarray(centers, dtype=np.float32)
        dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dists.append(float(np.linalg.norm(centers[i] - centers[j])))
        if not dists:
            return 0.0
        return float(np.mean(dists))

    @staticmethod
    def _tensor_class_center_distance(data, labels):
        if data is None or labels is None:
            return 0.0
        centers = []
        for label in torch.unique(labels):
            mask = labels == label
            if int(mask.sum().item()) < 2:
                continue
            centers.append(data[mask].detach().mean(dim=0).reshape(-1))
        if len(centers) <= 1:
            return 0.0
        centers = torch.stack(centers, dim=0)
        dists = torch.pdist(centers, p=2)
        if dists.numel() == 0:
            return 0.0
        return float(dists.mean().detach().cpu())

    def _resolve_seasonal_periods(self):
        raw = str(getattr(self.opt, "seasonal_periods", "auto")).strip().lower()
        seq_len = int(self.real_data.shape[1]) if self.real_data.ndim >= 2 else int(getattr(self.opt, "seq_len", 0))
        if seq_len <= 2:
            return []

        min_period = max(2, int(getattr(self.opt, "seasonal_min_period", 8)))
        max_period = int(getattr(self.opt, "seasonal_max_period", 0))
        if max_period <= 0:
            max_period = max(min_period, seq_len // 2)
        max_period = min(max_period, seq_len - 1)

        if raw and raw != "auto":
            periods = []
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    period = int(token)
                except ValueError:
                    continue
                if min_period <= period <= max_period:
                    periods.append(period)
            return sorted(set(periods))

        mean_series = self.real_data.mean(axis=2)
        avg_power = np.abs(np.fft.rfft(mean_series, axis=1)) ** 2
        avg_power = avg_power.mean(axis=0)
        candidate_idx = np.argsort(avg_power[1:])[::-1] + 1
        topk = max(1, int(getattr(self.opt, "seasonal_topk", 3)))
        periods = []
        for idx in candidate_idx:
            period = int(round(float(seq_len) / float(idx)))
            if period < min_period or period > max_period:
                continue
            if any(abs(period - existing) <= 1 for existing in periods):
                continue
            periods.append(period)
            if len(periods) >= topk:
                break
        return sorted(periods)

    def get_real_data(self, n=None, normalized=False):
        source = self.ori_data if normalized else self.real_data
        return self._to_numpy_batch(source, n=n)

    def get_real_labels(self, n=None):
        if self.real_labels is None:
            return None
        labels = np.asarray(self.real_labels, dtype=np.int64)
        if n is not None:
            labels = labels[: int(n)]
        return labels

    def _labels_to_condition(self, labels):
        if not self.use_class_conditioning or labels is None:
            return None
        return F.one_hot(labels.long(), num_classes=int(self.opt.num_classes)).float()

    def _sample_generation_labels(self, num_samples):
        if not self.use_class_conditioning:
            return None
        strategy = str(getattr(self.opt, "condition_sampling", "empirical")).lower()
        if strategy == "balanced":
            return np.random.randint(0, int(self.opt.num_classes), size=int(num_samples), dtype=np.int64)
        return np.random.choice(int(self.opt.num_classes), size=int(num_samples), p=self.label_probs).astype(np.int64)

    def _balanced_generation_labels(self, num_samples):
        if not self.use_class_conditioning:
            return None
        num_classes = max(1, int(self.opt.num_classes))
        labels = np.arange(num_classes, dtype=np.int64)
        repeats = int(np.ceil(float(num_samples) / float(num_classes)))
        samples = np.tile(labels, repeats)[: int(num_samples)].copy()
        np.random.shuffle(samples)
        return samples

    def _use_class_generator_match(self):
        if not self.use_class_conditioning:
            return False
        if not bool(getattr(self.opt, "class_generator_match", True)):
            return False
        if not hasattr(self, "netg"):
            return False
        return int(self.netg.num_generators) == int(self.opt.num_classes) and int(self.opt.num_classes) > 1

    def _router_targets(self, labels):
        if labels is None or not self._use_class_generator_match():
            return None
        return labels.long().clamp(min=0, max=max(0, int(self.netg.num_generators) - 1))

    def _sample_batch(self):
        batch_size = min(int(self.opt.batch_size), self.data_num)
        replace = batch_size > self.data_num
        indices = np.random.choice(self.data_num, size=batch_size, replace=replace)
        self.X0 = [self.ori_data[idx] for idx in indices]
        self.T = [self.ori_time[idx] for idx in indices]
        self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32, device=self.device)
        if self.real_labels is None:
            self.L = None
            self.C = None
        else:
            self.L = torch.tensor(self.real_labels[indices], dtype=torch.long, device=self.device)
            self.C = self._labels_to_condition(self.L)
        return batch_size

    def _to_data_scale(self, batch):
        return batch * self.max_val + self.min_val

    def _score_eval_metrics(self, paired_mae, paired_rmse, nn_mae, nn_rmse):
        metric_name = str(getattr(self.opt, "es_metric", "nn_combo"))
        mae_w = float(getattr(self.opt, "es_mae_weight", 0.5))
        if metric_name == "nn_rmse":
            return nn_rmse
        if metric_name == "nn_mae":
            return nn_mae
        if metric_name == "paired_rmse":
            return paired_rmse
        if metric_name == "paired_mae":
            return paired_mae
        return nn_rmse + mae_w * nn_mae

    def _eval_metrics_from_batches(self, real, fake):
        n = min(len(real), len(fake))
        if n <= 0:
            return {
                "n_eval": 0,
                "paired_mae": float("inf"),
                "paired_rmse": float("inf"),
                "nn_mae": float("inf"),
                "nn_rmse": float("inf"),
                "score": float("inf"),
            }

        real = real[:n]
        fake = fake[:n]
        paired_mae, paired_rmse = self._mae_rmse_np(real, fake)

        real_flat = real.reshape(n, -1)
        fake_flat = fake.reshape(n, -1)
        real_norm2 = np.sum(real_flat * real_flat, axis=1, keepdims=True).T
        fake_norm2 = np.sum(fake_flat * fake_flat, axis=1, keepdims=True)
        dist2 = fake_norm2 + real_norm2 - 2.0 * (fake_flat @ real_flat.T)
        dist2 = np.maximum(dist2, 0.0)
        nn_idx = np.argmin(dist2, axis=1)
        matched_real = real[nn_idx]
        nn_mae, nn_rmse = self._mae_rmse_np(matched_real, fake)
        score = self._score_eval_metrics(paired_mae, paired_rmse, nn_mae, nn_rmse)

        return {
            "n_eval": int(n),
            "paired_mae": float(paired_mae),
            "paired_rmse": float(paired_rmse),
            "nn_mae": float(nn_mae),
            "nn_rmse": float(nn_rmse),
            "score": float(score),
        }

    def _eval_metrics_by_label(self, real, real_labels, fake, fake_labels):
        real_labels = np.asarray(real_labels, dtype=np.int64).reshape(-1)
        fake_labels = np.asarray(fake_labels, dtype=np.int64).reshape(-1)
        label_set = sorted(set(real_labels.tolist()) & set(fake_labels.tolist()))
        rows = []
        for label in label_set:
            real_mask = real_labels == label
            fake_mask = fake_labels == label
            row = self._eval_metrics_from_batches(real[real_mask], fake[fake_mask])
            if row["n_eval"] > 0:
                rows.append(row)

        if not rows:
            return self._eval_metrics_from_batches(real, fake)

        return {
            "n_eval": int(sum(row["n_eval"] for row in rows)),
            "paired_mae": float(np.mean([row["paired_mae"] for row in rows])),
            "paired_rmse": float(np.mean([row["paired_rmse"] for row in rows])),
            "nn_mae": float(np.mean([row["nn_mae"] for row in rows])),
            "nn_rmse": float(np.mean([row["nn_rmse"] for row in rows])),
            "score": float(np.mean([row["score"] for row in rows])),
            "n_classes": int(len(rows)),
        }

    def _stage3_eval_metrics(self, eval_samples):
        if self.use_class_conditioning and self.real_labels is not None:
            eval_labels = self._balanced_generation_labels(eval_samples)
            fake_batch, fake_labels = self.generation(eval_samples, labels=eval_labels, return_labels=True)
            fake = self._to_numpy_batch(fake_batch)
            real = self.get_real_data(normalized=False)
            real_labels = self.get_real_labels()
            return self._eval_metrics_by_label(real, real_labels, fake, fake_labels)

        fake = self._to_numpy_batch(self.generation(eval_samples))
        real = self.get_real_data(n=len(fake), normalized=False)
        return self._eval_metrics_from_batches(real, fake)

    def _metric_es_dir(self):
        out_dir = os.path.join(self.opt.outf, self.opt.name, "metric_earlystop")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _train_diag_dir(self):
        out_dir = os.path.join(self.opt.outf, self.opt.name, "train", "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _collect_training_diagnostics(self, stage, step):
        row = {
            "stage": str(stage),
            "step": int(step),
            "joint_step": int(self._joint_step + 1) if str(stage) == "stage3_joint" else 0,
            "g_loss": self._safe_float(getattr(self, "err_g", None)),
            "d_loss": self._safe_float(getattr(self, "err_d", None)),
            "er_loss": self._safe_float(getattr(self, "err_er", None)),
            "s_loss": self._safe_float(getattr(self, "err_s", None)),
            "g_adv_group": self._safe_float(getattr(self, "err_g_adv_group", None)),
            "g_sup_group": self._safe_float(getattr(self, "err_g_sup_group", None)),
            "g_stat_group": self._safe_float(getattr(self, "err_g_stat_group", None)),
            "g_div_group": self._safe_float(getattr(self, "err_g_div_group", None)),
            "g_adv_latent": self._safe_float(getattr(self, "err_g_U", None)),
            "g_adv_embed": self._safe_float(getattr(self, "err_g_U_e", None)),
            "g_diversity": self._safe_float(getattr(self, "err_div", None)),
            "g_mode_seek": self._safe_float(getattr(self, "err_mode_seek", None)),
            "g_gate_balance": self._safe_float(getattr(self, "err_gate_bal", None)),
            "g_router": self._safe_float(getattr(self, "err_router", None)),
            "g_class_center": self._safe_float(getattr(self, "err_class_center", None)),
            "g_class_distance": self._safe_float(getattr(self, "err_class_distance", None)),
            "g_moment_std": self._safe_float(getattr(self, "err_g_V1", None)),
            "g_moment_mean": self._safe_float(getattr(self, "err_g_V2", None)),
            "g_ms": self._safe_float(getattr(self, "err_ms", None)),
            "g_freq": self._safe_float(getattr(self, "err_freq", None)),
            "g_trend": self._safe_float(getattr(self, "err_trend", None)),
            "g_seasonal_freq": self._safe_float(getattr(self, "err_seasonal_freq", None)),
            "g_seasonal_acf": self._safe_float(getattr(self, "err_seasonal_acf", None)),
            "g_swd": self._safe_float(getattr(self, "err_swd", None)),
            "g_roll": self._safe_float(getattr(self, "err_roll", None)),
            "g_compact": self._safe_float(getattr(self, "err_compact", None)),
            "d_real": self._safe_float(getattr(self, "err_d_real", None)),
            "d_fake": self._safe_float(getattr(self, "err_d_fake", None)),
            "d_fake_e": self._safe_float(getattr(self, "err_d_fake_e", None)),
            "batch_real_class_dist": self._safe_float(getattr(self, "batch_real_class_dist", 0.0), default=0.0),
            "batch_fake_class_dist": self._safe_float(getattr(self, "batch_fake_class_dist", 0.0), default=0.0),
            "global_real_class_dist": self._safe_float(self.real_class_distance, default=0.0),
        }
        if row["batch_real_class_dist"] > 1e-8:
            row["batch_fake_real_dist_ratio"] = row["batch_fake_class_dist"] / row["batch_real_class_dist"]
        elif row["global_real_class_dist"] > 1e-8:
            row["batch_fake_real_dist_ratio"] = row["batch_fake_class_dist"] / row["global_real_class_dist"]
        else:
            row["batch_fake_real_dist_ratio"] = 0.0
        for idx, period in enumerate(self.seasonal_periods):
            row[f"seasonal_period_{idx}"] = int(period)

        gate_weights = getattr(self, "gate_weights", None)
        if gate_weights is not None and torch.is_tensor(gate_weights) and gate_weights.dim() == 2:
            gate_mean = gate_weights.detach().mean(dim=0).cpu().numpy().tolist()
            for idx, value in enumerate(gate_mean):
                row[f"gate_mean_{idx}"] = float(value)

        selected_ids = getattr(self, "selected_ids", None)
        if selected_ids is not None and torch.is_tensor(selected_ids):
            route_hist = torch.bincount(selected_ids.detach().cpu(), minlength=int(self.netg.num_generators)).float()
            denom = max(1.0, float(route_hist.sum().item()))
            for idx, value in enumerate((route_hist / denom).tolist()):
                row[f"route_frac_{idx}"] = float(value)

        return row

    def _print_training_diagnostics(self, row):
        gate_cols = [key for key in sorted(row) if key.startswith("gate_mean_")]
        gate_text = ", ".join(f"{row[key]:.3f}" for key in gate_cols[:4])
        print(
            "  [Diag] "
            f"{row['stage']} step={row['step']} "
            f"adv={row['g_adv_group']:.4f} sup={row['g_sup_group']:.4f} "
            f"stat={row['g_stat_group']:.4f} div={row['g_div_group']:.4f} "
            f"msk={row.get('g_mode_seek', np.nan):.4f} "
            f"router={row['g_router']:.4f} cls_center={row['g_class_center']:.4f} "
            f"cls_dist={row['g_class_distance']:.4f} fake/real_dist={row['batch_fake_real_dist_ratio']:.4f}"
            + (f" gate=[{gate_text}]" if gate_text else "")
        )

    def _save_training_diagnostics(self, rows):
        if not rows:
            return
        diag_dir = self._train_diag_dir()
        json_path = os.path.join(diag_dir, "train_diagnostics.json")
        csv_path = os.path.join(diag_dir, "train_diagnostics.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        preferred = [
            "stage",
            "step",
            "joint_step",
            "g_loss",
            "d_loss",
            "er_loss",
            "s_loss",
            "g_adv_group",
            "g_sup_group",
            "g_stat_group",
            "g_div_group",
            "g_mode_seek",
            "g_router",
            "g_class_center",
            "g_class_distance",
            "batch_real_class_dist",
            "batch_fake_class_dist",
            "batch_fake_real_dist_ratio",
            "global_real_class_dist",
        ]
        all_keys = []
        seen = set()
        for key in preferred:
            if any(key in row for row in rows):
                all_keys.append(key)
                seen.add(key)
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    all_keys.append(key)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _save_metric_best_checkpoint(self, step, metrics, score_ema):
        es_dir = self._metric_es_dir()
        ckpt = {
            "step": int(step),
            "metrics": metrics,
            "score_ema": float(score_ema),
            "netE": self.nete.state_dict(),
            "netR": self.netr.state_dict(),
            "netG": self.netg.state_dict(),
            "netD": self.netd.state_dict(),
            "netS": self.nets.state_dict(),
        }
        torch.save(ckpt, os.path.join(es_dir, "best_metric.pth"))
        with open(os.path.join(es_dir, "best_metric.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": int(step),
                    "score_ema": float(score_ema),
                    **metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _load_metric_best_checkpoint(self):
        ckpt_path = os.path.join(self.opt.outf, self.opt.name, "metric_earlystop", "best_metric.pth")
        if not os.path.exists(ckpt_path):
            return False
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.nete.load_state_dict(ckpt["netE"])
        self.netr.load_state_dict(ckpt["netR"])
        self.netg.load_state_dict(ckpt["netG"])
        self.netd.load_state_dict(ckpt["netD"])
        self.nets.load_state_dict(ckpt["netS"])
        return True

    def train_one_iter_er(self):
        self.nete.train()
        self.netr.train()
        self._sample_batch()
        self.optimize_params_er()

    def train_one_iter_er_(self):
        self.nete.train()
        self.netr.train()
        self._sample_batch()
        self.optimize_params_er_()

    def train_one_iter_s(self):
        self.nets.train()
        self._sample_batch()
        self.optimize_params_s()

    def train_one_iter_g(self):
        self.netg.train()
        batch_size = self._sample_batch()
        self.Z = random_generator(batch_size, self.opt.z_dim, self.T, self.max_seq_len)
        self.optimize_params_g()

    def train_one_iter_d(self):
        self.netd.train()
        batch_size = self._sample_batch()
        self.Z = random_generator(batch_size, self.opt.z_dim, self.T, self.max_seq_len)
        self.optimize_params_d()

    def train(self):
        stage1_iters = int(getattr(self.opt, "stage1_iters", self.opt.iteration))
        stage2_iters = int(getattr(self.opt, "stage2_iters", self.opt.iteration))
        stage3_iters = int(getattr(self.opt, "stage3_iters", self.opt.iteration))
        joint_g_steps = max(1, int(getattr(self.opt, "joint_g_steps", 2)))
        log_interval = max(1, int(getattr(self.opt, "print_freq", 1000)))
        self._joint_total_steps = stage3_iters
        use_metric_earlystop = bool(getattr(self.opt, "use_metric_earlystop", False))
        train_diag_rows = []

        if use_metric_earlystop:
            es_eval_interval = int(getattr(self.opt, "es_eval_interval", 200))
            es_patience = int(getattr(self.opt, "es_patience", 8))
            es_min_delta = float(getattr(self.opt, "es_min_delta", 1e-3))
            es_eval_samples = int(getattr(self.opt, "es_eval_samples", 256))
            es_warmup_steps = int(getattr(self.opt, "es_warmup_steps", 1000))
            es_ema_beta = float(getattr(self.opt, "es_ema_beta", 0.6))
            es_metric = str(getattr(self.opt, "es_metric", "nn_combo"))
            print(
                "[MetricEarlyStop] enabled "
                f"metric={es_metric}, eval_interval={es_eval_interval}, patience={es_patience}, "
                f"min_delta={es_min_delta}, eval_samples={es_eval_samples}, warmup={es_warmup_steps}, ema_beta={es_ema_beta}"
            )
        else:
            es_eval_interval = 0
            es_patience = 0
            es_min_delta = 0.0
            es_eval_samples = 0
            es_warmup_steps = 0
            es_ema_beta = 0.0

        print(f"[Stage 1/3] Autoencoder pretrain for {stage1_iters} iterations")
        for step in range(stage1_iters):
            self.train_one_iter_er()
            if step % max(1, stage1_iters // 10) == 0:
                print(f"  ER step {step}/{stage1_iters}, loss={float(self.err_er.detach().cpu()):.6f}")
            if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == stage1_iters:
                train_diag_rows.append(self._collect_training_diagnostics("stage1_er", step + 1))

        print(f"[Stage 2/3] Supervisor pretrain for {stage2_iters} iterations")
        for step in range(stage2_iters):
            self.train_one_iter_s()
            if step % max(1, stage2_iters // 10) == 0:
                print(f"  S step {step}/{stage2_iters}, loss={float(self.err_s.detach().cpu()):.6f}")
            if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == stage2_iters:
                train_diag_rows.append(self._collect_training_diagnostics("stage2_s", step + 1))

        print(f"[Stage 3/3] Joint training for {stage3_iters} iterations")
        best_score_ema = float("inf")
        ema_score = None
        bad_count = 0
        stopped_step = stage3_iters
        es_log_rows = []
        for step in range(stage3_iters):
            self._joint_step = step
            for _ in range(joint_g_steps):
                self.train_one_iter_g()
                self.train_one_iter_er_()
            self.train_one_iter_d()
            if step % max(1, stage3_iters // 10) == 0:
                g_loss = float(self.err_g.detach().cpu()) if hasattr(self, "err_g") else -1.0
                d_loss = float(self.err_d.detach().cpu()) if hasattr(self, "err_d") else -1.0
                print(f"  Joint step {step}/{stage3_iters}, g={g_loss:.6f}, d={d_loss:.6f}")
            if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == stage3_iters:
                diag_row = self._collect_training_diagnostics("stage3_joint", step + 1)
                train_diag_rows.append(diag_row)
                self._print_training_diagnostics(diag_row)

            if use_metric_earlystop and ((step + 1) % es_eval_interval == 0 or (step + 1) == stage3_iters):
                eval_row = self._stage3_eval_metrics(es_eval_samples)
                raw_score = float(eval_row["score"])
                if ema_score is None:
                    ema_score = raw_score
                else:
                    ema_score = es_ema_beta * ema_score + (1.0 - es_ema_beta) * raw_score

                eval_row["step"] = int(step + 1)
                eval_row["score_ema"] = float(ema_score)
                es_log_rows.append(eval_row)
                print(
                    "  [MetricEval] "
                    f"step={step+1} score={raw_score:.6f} ema={ema_score:.6f} "
                    f"nn_rmse={eval_row['nn_rmse']:.6f} nn_mae={eval_row['nn_mae']:.6f} "
                    f"paired_rmse={eval_row['paired_rmse']:.6f} paired_mae={eval_row['paired_mae']:.6f}"
                )

                improved = ema_score < (best_score_ema - es_min_delta)
                if np.isinf(best_score_ema):
                    improved = True
                if improved:
                    best_score_ema = float(ema_score)
                    bad_count = 0
                    self._save_metric_best_checkpoint(step + 1, eval_row, best_score_ema)
                    print(f"  [MetricEarlyStop] new best ema={best_score_ema:.6f} at step {step+1}")
                elif (step + 1) >= es_warmup_steps:
                    bad_count += 1
                    print(f"  [MetricEarlyStop] no improvement {bad_count}/{es_patience}")
                    if bad_count >= es_patience:
                        stopped_step = int(step + 1)
                        print(f"  [MetricEarlyStop] triggered at stage-3 step {stopped_step}")
                        break

        if use_metric_earlystop:
            es_dir = self._metric_es_dir()
            if es_log_rows:
                with open(os.path.join(es_dir, "metric_score_log.csv"), "w", encoding="utf-8") as f:
                    f.write("step,score,score_ema,nn_rmse,nn_mae,paired_rmse,paired_mae,n_eval\n")
                    for r in es_log_rows:
                        f.write(
                            f"{r['step']},{r['score']:.8f},{r['score_ema']:.8f},{r['nn_rmse']:.8f},{r['nn_mae']:.8f},"
                            f"{r['paired_rmse']:.8f},{r['paired_mae']:.8f},{r['n_eval']}\n"
                        )
                with open(os.path.join(es_dir, "metric_score_log.json"), "w", encoding="utf-8") as f:
                    json.dump(es_log_rows, f, ensure_ascii=False, indent=2)

            restored = self._load_metric_best_checkpoint()
            print(f"[MetricEarlyStop] restore best checkpoint: {restored}")

        self.apply_pruning_if_needed()

        post_prune_finetune_iters = int(getattr(self.opt, "post_prune_finetune_iters", 0))
        if post_prune_finetune_iters > 0:
            print(f"[Pruning] Fine-tuning {post_prune_finetune_iters} iterations")
            for _ in range(post_prune_finetune_iters):
                self.train_one_iter_g()
                self.train_one_iter_er_()
                self.train_one_iter_d()

        save_step = int(stopped_step if use_metric_earlystop else stage3_iters)
        last_joint_logged = 0
        for row in train_diag_rows:
            if row.get("stage") == "stage3_joint":
                last_joint_logged = max(last_joint_logged, int(row.get("step", 0)))
        if stage3_iters > 0 and save_step > 0 and last_joint_logged != save_step:
            diag_row = self._collect_training_diagnostics("stage3_joint", save_step)
            train_diag_rows.append(diag_row)
            self._print_training_diagnostics(diag_row)
        self._save_training_diagnostics(train_diag_rows)
        self.save_weights(save_step)
        self.generated_data, self.generated_labels = self.generation(self.opt.generate_num, return_labels=True)
        print("Finish synthetic data generation")

    @staticmethod
    def _warmup_factor(step, warmup_steps):
        warmup_steps = int(warmup_steps)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    def _pick_generator_ids(self, gate_weights, allow_soft=True):
        mode = getattr(self.opt, "mode_selection", "hybrid")
        if mode == "soft" and allow_soft:
            return None
        if mode == "soft" and not allow_soft:
            return torch.argmax(gate_weights, dim=1)

        batch_size = gate_weights.size(0)
        if mode == "disc":
            return torch.argmax(gate_weights, dim=1)

        # hybrid: half argmax + half multinomial sampling
        disc_part = torch.argmax(gate_weights, dim=1)
        prob_part = torch.multinomial(gate_weights, num_samples=1).squeeze(1)
        mask = torch.rand(batch_size, device=gate_weights.device) < 0.5
        return torch.where(mask, disc_part, prob_part)

    def _route_generator_outputs(self, all_outputs, gate_weights, hard=False, selected_ids=None):
        if not hard:
            return torch.einsum("gbth,bg->bth", all_outputs, gate_weights), selected_ids
        if selected_ids is None:
            selected_ids = self._pick_generator_ids(gate_weights, allow_soft=False)
        hard_mask = F.one_hot(selected_ids, num_classes=gate_weights.size(1)).type_as(gate_weights)
        st_weights = hard_mask - gate_weights.detach() + gate_weights
        routed = torch.einsum("gbth,bg->bth", all_outputs, st_weights)
        return routed, selected_ids

    def generation(self, num_samples, g_idx=None, labels=None, return_labels=False, mean=0, std=1):
        del mean, std
        if num_samples == 0:
            return ([], np.asarray([], dtype=np.int64)) if return_labels else []

        generated_data = []
        generated_labels = []
        inference_batch_size = self.opt.batch_size
        labels_np = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_np is not None and len(labels_np) < num_samples:
            raise ValueError("Provided labels are fewer than num_samples.")

        for i in range(0, num_samples, inference_batch_size):
            current_size = min(inference_batch_size, num_samples - i)
            current_T = np.random.choice(self.ori_time, size=current_size, replace=True)
            z_curr = random_generator(current_size, self.opt.z_dim, current_T, self.max_seq_len)
            z_curr_tensor = torch.tensor(np.asarray(z_curr), dtype=torch.float32, device=self.device)
            if labels_np is None:
                current_labels = self._sample_generation_labels(current_size)
            else:
                current_labels = labels_np[i : i + current_size]
            cond = None
            if current_labels is not None:
                label_tensor = torch.tensor(current_labels, dtype=torch.long, device=self.device)
                cond = self._labels_to_condition(label_tensor)

            with torch.no_grad():
                if g_idx is not None:
                    E_hat = self.netg.forward_single(z_curr_tensor, g_idx, condition=cond)
                else:
                    mixed_E, gate_weights, all_E = self.netg(z_curr_tensor, condition=cond, return_details=True)
                    if current_labels is not None and self._use_class_generator_match():
                        selected_ids = torch.tensor(current_labels, dtype=torch.long, device=self.device)
                    else:
                        selected_ids = self._pick_generator_ids(gate_weights)
                    if selected_ids is None:
                        E_hat = mixed_E
                    else:
                        all_e_b = all_E.permute(1, 0, 2, 3)  # [B, G, T, H]
                        gather_idx = selected_ids.view(-1, 1, 1, 1).expand(-1, 1, all_e_b.size(2), all_e_b.size(3))
                        E_hat = torch.gather(all_e_b, dim=1, index=gather_idx).squeeze(1)

                H_hat = self.nets(E_hat)
                output_batch = self.netr(H_hat).cpu().numpy()

            for j in range(current_size):
                seq_len = int(current_T[j])
                sample_seg = output_batch[j, :seq_len, :]
                sample_renorm = self._to_data_scale(sample_seg)
                generated_data.append(sample_renorm)
                if current_labels is not None:
                    generated_labels.append(int(current_labels[j]))

        if return_labels:
            return generated_data, np.asarray(generated_labels, dtype=np.int64)
        return generated_data

    def apply_pruning_if_needed(self):
        prune_amount = float(getattr(self.opt, "prune_amount", 0.0))
        if prune_amount <= 0.0:
            return

        prune_mode = getattr(self.opt, "prune_mode", "hybrid")
        print(f"[Pruning] mode={prune_mode}, amount={prune_amount}")

        if prune_mode in ["generators", "hybrid"]:
            self._prune_generators(prune_amount)

        if prune_mode in ["weights", "hybrid"]:
            self._prune_linear_weights(prune_amount)

    def _prune_generators(self, prune_amount):
        current_g = int(self.netg.num_generators)
        if current_g <= 1:
            return

        keep_g = max(1, int(round(current_g * (1.0 - prune_amount))))
        if keep_g >= current_g:
            return

        score = torch.zeros(current_g, device=self.device)
        calib_steps = min(10, max(2, self.data_num // max(1, self.opt.batch_size)))

        with torch.no_grad():
            for _ in range(calib_steps):
                batch_size = min(int(self.opt.batch_size), self.data_num)
                replace = batch_size > self.data_num
                indices = np.random.choice(self.data_num, size=batch_size, replace=replace)
                t_mb = [self.ori_time[idx] for idx in indices]
                labels_mb = None if self.real_labels is None else torch.tensor(self.real_labels[indices], dtype=torch.long, device=self.device)
                cond_mb = self._labels_to_condition(labels_mb)
                Z = random_generator(batch_size, self.opt.z_dim, t_mb, self.max_seq_len)
                Z = torch.tensor(np.asarray(Z), dtype=torch.float32, device=self.device)
                _, gate_weights, _ = self.netg(Z, condition=cond_mb, return_details=True)
                score += gate_weights.mean(dim=0)

        keep_indices = torch.topk(score, k=keep_g, largest=True).indices.detach().cpu().tolist()
        self.netg.prune_generators(keep_indices)
        print(f"[Pruning] generators kept={keep_indices}, now num_generators={self.netg.num_generators}")

    def _prune_linear_weights(self, prune_amount):
        modules = [self.netg, self.nets, self.netr, self.netd]
        pruned = 0
        for root in modules:
            for m in root.modules():
                if isinstance(m, nn.Linear):
                    prune.l1_unstructured(m, name="weight", amount=prune_amount)
                    prune.remove(m, "weight")
                    pruned += 1
        print(f"[Pruning] linear layers pruned={pruned}")


class TimeGAN(BaseModel):
    """TimeGAN with TransGRU multi-generator extension."""

    @property
    def name(self):
        return "TimeGAN"

    def __init__(self, opt, ori_data, ori_labels=None):
        super(TimeGAN, self).__init__(opt, ori_data, ori_labels=ori_labels)

        self.nete = Encoder(self.opt).to(self.device)
        self.netr = Recovery(self.opt).to(self.device)
        self.netg = MultiGeneratorEnsemble(self.opt).to(self.device)
        self.netd = Discriminator(self.opt).to(self.device)
        self.nets = Supervisor(self.opt).to(self.device)

        self._load_weights()

        self.l_mse = nn.MSELoss()
        self.disc_use_logits = bool(getattr(self.opt, "disc_use_logits", True))
        self.l_bce = nn.BCEWithLogitsLoss() if self.disc_use_logits else nn.BCELoss()
        self.real_label_smooth = float(getattr(self.opt, "label_smooth_real", 1.0))
        self.fake_label_smooth = float(getattr(self.opt, "label_smooth_fake", 0.0))

        if self.opt.isTrain:
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()

            self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.e_lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.r_lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.g_lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.d_lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.s_lr, betas=(self.opt.beta1, 0.999))

    def forward_e(self):
        self.H = self.nete(self.X)

    def forward_er(self):
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        self.Z = torch.tensor(np.asarray(self.Z), dtype=torch.float32, device=self.device)
        mixed_e, self.gate_weights, self.all_E_hat = self.netg(self.Z, condition=self.C, return_details=True)
        use_hard = bool(getattr(self.opt, "train_hard_routing", True))
        if use_hard and self.netg.num_generators > 1:
            route_targets = self._router_targets(self.L)
            self.E_hat, self.selected_ids = self._route_generator_outputs(
                self.all_E_hat,
                self.gate_weights,
                hard=True,
                selected_ids=route_targets,
            )
        else:
            self.E_hat = mixed_e
            self.selected_ids = None
        self.Z_mode_seek = None
        self.H_hat_mode_seek = None
        if float(getattr(self.opt, "lambda_mode_seeking", 0.0)) > 0.0 and int(self.netg.num_generators) == 1:
            alt_z = random_generator(int(self.Z.size(0)), self.opt.z_dim, self.T, self.max_seq_len)
            self.Z_mode_seek = torch.tensor(np.asarray(alt_z), dtype=torch.float32, device=self.device)
            alt_e = self.netg(self.Z_mode_seek, condition=self.C, return_details=False)
            self.H_hat_mode_seek = self.nets(alt_e)

    def _adv_factor(self):
        return self._warmup_factor(self._joint_step, getattr(self.opt, "adv_warmup_steps", 0))

    def _cross_factor(self):
        return self._warmup_factor(self._joint_step, getattr(self.opt, "cross_warmup_steps", 0))

    def _label(self, ref, is_real):
        value = self.real_label_smooth if is_real else self.fake_label_smooth
        return torch.full_like(ref, fill_value=float(value))

    def _add_instance_noise(self, x):
        base_std = float(getattr(self.opt, "instance_noise_std", 0.0))
        if base_std <= 0.0:
            return x
        total = max(1, int(self._joint_total_steps))
        decay = 1.0 - min(1.0, float(self._joint_step) / float(total))
        std = base_std * decay
        if std <= 1e-8:
            return x
        return x + std * torch.randn_like(x)

    def _disc_forward(self, x, condition=None):
        x = self._add_instance_noise(x)
        return self.netd(x, condition=condition, sigmoid=not self.disc_use_logits)

    def _classwise_reduce(self, fn, real, fake, labels=None):
        use_classwise = bool(getattr(self.opt, "classwise_stat_loss", True))
        if labels is None or not use_classwise:
            return fn(real, fake)

        unique_labels = torch.unique(labels)
        outputs = []
        for label in unique_labels:
            mask = labels == label
            if int(mask.sum().item()) < 2:
                continue
            outputs.append(fn(real[mask], fake[mask]))

        if not outputs:
            return fn(real, fake)

        first = outputs[0]
        if isinstance(first, tuple):
            reduced = []
            for idx in range(len(first)):
                reduced.append(torch.stack([out[idx] for out in outputs]).mean())
            return tuple(reduced)
        return torch.stack(outputs).mean()

    def _moment_loss(self, real, fake, labels=None):
        def _loss_fn(real_local, fake_local):
            real_std = torch.std(real_local, dim=(0, 1), unbiased=False)
            fake_std = torch.std(fake_local, dim=(0, 1), unbiased=False)
            real_mean = torch.mean(real_local, dim=(0, 1))
            fake_mean = torch.mean(fake_local, dim=(0, 1))
            v_std = torch.mean(torch.abs(fake_std - real_std))
            v_mean = torch.mean(torch.abs(fake_mean - real_mean))
            return v_std, v_mean

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _multiscale_loss(self, real, fake, labels=None):
        scales = list(getattr(self.opt, "ms_scales", [2, 4, 8]))

        def _loss_fn(real_local, fake_local):
            loss_terms = []
            real_t = real_local.permute(0, 2, 1)
            fake_t = fake_local.permute(0, 2, 1)
            for s in scales:
                s = int(s)
                if s <= 1 or real_t.size(-1) < s:
                    continue
                pr = F.avg_pool1d(real_t, kernel_size=s, stride=s)
                pf = F.avg_pool1d(fake_t, kernel_size=s, stride=s)
                loss_terms.append(torch.mean(torch.abs(pr - pf)))
            if not loss_terms:
                return real_local.new_zeros(())
            return torch.stack(loss_terms).mean()

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _frequency_loss(self, real, fake, labels=None):
        def _loss_fn(real_local, fake_local):
            fr = torch.log1p(torch.abs(torch.fft.rfft(real_local, dim=1)))
            ff = torch.log1p(torch.abs(torch.fft.rfft(fake_local, dim=1)))
            return torch.mean(torch.abs(fr.mean(dim=0) - ff.mean(dim=0)))

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _trend_loss(self, real, fake, labels=None):
        def _loss_fn(real_local, fake_local):
            dr = real_local[:, 1:, :] - real_local[:, :-1, :]
            df = fake_local[:, 1:, :] - fake_local[:, :-1, :]
            mean_gap = torch.mean(torch.abs(dr.mean(dim=(0, 1)) - df.mean(dim=(0, 1))))
            std_gap = torch.mean(
                torch.abs(torch.std(dr, dim=(0, 1), unbiased=False) - torch.std(df, dim=(0, 1), unbiased=False))
            )
            return mean_gap + std_gap

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _seasonal_frequency_loss(self, real, fake, labels=None):
        periods = list(getattr(self, "seasonal_periods", []))

        def _loss_fn(real_local, fake_local):
            if not periods:
                return real_local.new_zeros(())
            fr = torch.log1p(torch.abs(torch.fft.rfft(real_local, dim=1)))
            ff = torch.log1p(torch.abs(torch.fft.rfft(fake_local, dim=1)))
            mean_r = fr.mean(dim=0)
            mean_f = ff.mean(dim=0)
            std_r = torch.std(fr, dim=0, unbiased=False)
            std_f = torch.std(ff, dim=0, unbiased=False)
            losses = []
            for period in periods:
                bin_idx = int(round(float(real_local.size(1)) / float(period)))
                if bin_idx < 1 or bin_idx >= mean_r.size(0):
                    continue
                losses.append(F.smooth_l1_loss(mean_f[bin_idx], mean_r[bin_idx]))
                losses.append(F.smooth_l1_loss(std_f[bin_idx], std_r[bin_idx]))
            if not losses:
                return real_local.new_zeros(())
            return torch.stack(losses).mean()

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _seasonal_acf_loss(self, real, fake, labels=None):
        periods = list(getattr(self, "seasonal_periods", []))
        harmonics = max(1, int(getattr(self.opt, "seasonal_harmonics", 2)))

        def _loss_fn(real_local, fake_local):
            if not periods:
                return real_local.new_zeros(())
            losses = []
            for period in periods:
                for mult in range(1, harmonics + 1):
                    lag = int(period) * int(mult)
                    if lag < 1 or lag >= real_local.size(1):
                        continue
                    acf_real = self._lag_autocorr_torch(real_local, lag)
                    acf_fake = self._lag_autocorr_torch(fake_local, lag)
                    losses.append(F.smooth_l1_loss(acf_fake, acf_real))
            if not losses:
                return real_local.new_zeros(())
            return torch.stack(losses).mean()

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    def _class_structure_loss(self, real, fake, labels=None):
        zero = real.new_zeros(())
        if labels is None:
            return zero, zero

        real_centers = []
        fake_centers = []
        for label in torch.unique(labels):
            mask = labels == label
            if int(mask.sum().item()) < 2:
                continue
            real_centers.append(real[mask].mean(dim=0).reshape(-1))
            fake_centers.append(fake[mask].mean(dim=0).reshape(-1))

        if not real_centers:
            return zero, zero

        real_centers = torch.stack(real_centers, dim=0)
        fake_centers = torch.stack(fake_centers, dim=0)
        loss_center = F.smooth_l1_loss(fake_centers, real_centers)

        if real_centers.size(0) <= 1:
            return loss_center, zero

        real_dist = torch.cdist(real_centers, real_centers, p=2)
        fake_dist = torch.cdist(fake_centers, fake_centers, p=2)
        pair_mask = ~torch.eye(real_centers.size(0), dtype=torch.bool, device=real_centers.device)
        loss_distance = F.smooth_l1_loss(fake_dist[pair_mask], real_dist[pair_mask])
        return loss_center, loss_distance

    @staticmethod
    def _lag_autocorr_torch(x, lag):
        # x: [B, T, D]
        if lag >= x.size(1):
            return torch.zeros((x.size(2),), device=x.device, dtype=x.dtype)
        x0 = x[:, :-lag, :]
        x1 = x[:, lag:, :]
        x0c = x0 - x0.mean(dim=1, keepdim=True)
        x1c = x1 - x1.mean(dim=1, keepdim=True)
        num = (x0c * x1c).mean(dim=1)
        den = (x0c.pow(2).mean(dim=1).sqrt() * x1c.pow(2).mean(dim=1).sqrt()) + 1e-6
        return (num / den).mean(dim=0)

    def _rollout_len(self, seq_len):
        seq_len = int(seq_len)
        if seq_len <= 2:
            return max(1, seq_len)
        max_roll = seq_len
        cap = int(getattr(self.opt, "rollout_cap", 0))
        if cap > 0:
            max_roll = min(max_roll, cap)
        start = int(getattr(self.opt, "rollout_start", max_roll))
        start = max(2, min(start, max_roll))
        ramp = int(getattr(self.opt, "rollout_ramp_steps", 0))
        if ramp <= 0:
            return max_roll
        prog = min(1.0, float(self._joint_step + 1) / float(max(1, ramp)))
        roll = int(round(start + (max_roll - start) * prog))
        return max(2, min(roll, max_roll))

    def _rollout_reg_loss(self, real_seg, fake_seg):
        # SigCWGAN-style rollout regularization on local segments.
        loss_level_std = F.smooth_l1_loss(
            torch.std(fake_seg, dim=0, unbiased=False),
            torch.std(real_seg, dim=0, unbiased=False),
        )

        real_diff = real_seg[:, 1:, :] - real_seg[:, :-1, :]
        fake_diff = fake_seg[:, 1:, :] - fake_seg[:, :-1, :]
        loss_diff_mean = F.smooth_l1_loss(fake_diff.mean(dim=(0, 1)), real_diff.mean(dim=(0, 1)))
        loss_diff_std = F.smooth_l1_loss(
            torch.std(fake_diff, dim=(0, 1), unbiased=False),
            torch.std(real_diff, dim=(0, 1), unbiased=False),
        )

        acf_terms = []
        for lag in list(getattr(self.opt, "rollout_acf_lags", [1, 2, 4, 8])):
            lag = int(lag)
            if lag < 1 or lag >= real_diff.size(1):
                continue
            acf_real = self._lag_autocorr_torch(real_diff, lag)
            acf_fake = self._lag_autocorr_torch(fake_diff, lag)
            acf_terms.append(F.smooth_l1_loss(acf_fake, acf_real))
        loss_acf = torch.stack(acf_terms).mean() if acf_terms else real_seg.new_zeros(())

        self.err_roll_level_std = loss_level_std
        self.err_roll_diff_mean = loss_diff_mean
        self.err_roll_diff_std = loss_diff_std
        self.err_roll_acf = loss_acf

        return (
            float(getattr(self.opt, "w_rollout_level_std", 0.35)) * loss_level_std
            + float(getattr(self.opt, "w_rollout_diff_mean", 0.15)) * loss_diff_mean
            + float(getattr(self.opt, "w_rollout_diff_std", 0.30)) * loss_diff_std
            + float(getattr(self.opt, "w_rollout_acf", 0.20)) * loss_acf
        )

    def _compact_stat_loss(self, real, fake, labels=None):
        def _loss_fn(real_local, fake_local):
            level_mean = F.smooth_l1_loss(fake_local.mean(dim=(0, 1)), real_local.mean(dim=(0, 1)))
            level_std = F.smooth_l1_loss(
                torch.std(fake_local, dim=(0, 1), unbiased=False),
                torch.std(real_local, dim=(0, 1), unbiased=False),
            )
            loss_level = 0.5 * (level_mean + level_std)

            real_diff = real_local[:, 1:, :] - real_local[:, :-1, :]
            fake_diff = fake_local[:, 1:, :] - fake_local[:, :-1, :]
            diff_mean = F.smooth_l1_loss(fake_diff.mean(dim=(0, 1)), real_diff.mean(dim=(0, 1)))
            diff_std = F.smooth_l1_loss(
                torch.std(fake_diff, dim=(0, 1), unbiased=False),
                torch.std(real_diff, dim=(0, 1), unbiased=False),
            )
            loss_diff = 0.5 * (diff_mean + diff_std)

            acf_terms = []
            for lag in list(getattr(self.opt, "compact_acf_lags", [1, 2, 4, 8])):
                lag = int(lag)
                if lag < 1 or lag >= real_diff.size(1):
                    continue
                acf_real = self._lag_autocorr_torch(real_diff, lag)
                acf_fake = self._lag_autocorr_torch(fake_diff, lag)
                acf_terms.append(F.smooth_l1_loss(acf_fake, acf_real))
            loss_acf = torch.stack(acf_terms).mean() if acf_terms else real_local.new_zeros(())
            return loss_level, loss_diff, loss_acf

        loss_level, loss_diff, loss_acf = self._classwise_reduce(_loss_fn, real, fake, labels=labels)
        self.err_compact_level = loss_level
        self.err_compact_diff = loss_diff
        self.err_compact_acf = loss_acf
        return (
            float(getattr(self.opt, "w_compact_level", 0.35)) * loss_level
            + float(getattr(self.opt, "w_compact_diff", 0.35)) * loss_diff
            + float(getattr(self.opt, "w_compact_acf", 0.30)) * loss_acf
        )

    def _sliced_wasserstein_loss(self, real, fake, num_projections=64, labels=None):
        def _loss_fn(real_local, fake_local):
            batch = min(real_local.size(0), fake_local.size(0))
            if batch <= 1:
                return real_local.new_zeros(())

            xr = real_local[:batch].reshape(batch, -1)
            xf = fake_local[:batch].reshape(batch, -1)
            proj = max(1, int(num_projections))
            directions = torch.randn(proj, xr.size(1), device=xr.device, dtype=xr.dtype)
            directions = F.normalize(directions, p=2, dim=1)
            pr = xr @ directions.t()
            pf = xf @ directions.t()
            pr_sorted, _ = torch.sort(pr, dim=0)
            pf_sorted, _ = torch.sort(pf, dim=0)
            return torch.mean(torch.abs(pr_sorted - pf_sorted))

        return self._classwise_reduce(_loss_fn, real, fake, labels=labels)

    @staticmethod
    def _gate_balance_loss(gate_weights):
        if gate_weights.size(1) <= 1:
            return gate_weights.new_zeros(())
        p = gate_weights.mean(dim=0)
        uniform = torch.full_like(p, 1.0 / p.numel())
        return torch.sum(p * torch.log((p + 1e-8) / (uniform + 1e-8)))

    def _mode_seeking_loss(self, z_a, z_b, out_a, out_b):
        eps = float(getattr(self.opt, "mode_seek_eps", 1e-4))
        z_dist = torch.mean(torch.abs(z_a - z_b), dim=(1, 2))
        out_dist = torch.mean(torch.abs(out_a - out_b), dim=(1, 2))
        return torch.mean(z_dist / (out_dist + eps))

    def _router_supervision_loss(self, gate_weights, labels):
        targets = self._router_targets(labels)
        if targets is None:
            return gate_weights.new_zeros(())
        return F.nll_loss(torch.log(gate_weights.clamp_min(1e-8)), targets)

    def _clip_grad(self, modules):
        clip = float(getattr(self.opt, "grad_clip", 0.0))
        if clip <= 0.0:
            return
        params = []
        for m in modules:
            params.extend([p for p in m.parameters() if p.grad is not None])
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)

    def forward_dg(self):
        self.Y_fake = self._disc_forward(self.H_hat, condition=self.C)
        self.Y_fake_e = self._disc_forward(self.E_hat, condition=self.C)

    def forward_rg(self):
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        self.H_supervise = self.nets(self.H)

    def forward_sg(self):
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        self.Y_real = self._disc_forward(self.H, condition=self.C)
        self.Y_fake = self._disc_forward(self.H_hat, condition=self.C)
        self.Y_fake_e = self._disc_forward(self.E_hat, condition=self.C)

    def backward_er(self):
        err_l2 = self.l_mse(self.X_tilde, self.X)
        err_l1 = F.l1_loss(self.X_tilde, self.X)
        self.err_er = err_l2 + self.opt.lambda_recon_l1 * err_l1
        self.err_er.backward(retain_graph=True)

    def backward_er_(self):
        self.err_er_ = self.l_mse(self.X_tilde, self.X)
        self.err_er_l1 = F.l1_loss(self.X_tilde, self.X)
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_er = (
            self.opt.w_e0 * torch.sqrt(self.err_er_ + 1e-8)
            + self.opt.w_es * self.err_s
            + self.opt.lambda_recon_l1 * self.err_er_l1
        )
        self.err_er.backward(retain_graph=True)

    def backward_g(self):
        self.err_g_U = self.l_bce(self.Y_fake, self._label(self.Y_fake, is_real=True))
        self.err_g_U_e = self.l_bce(self.Y_fake_e, self._label(self.Y_fake_e, is_real=True))
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.batch_real_class_dist = self._tensor_class_center_distance(self.X, self.L)
        self.batch_fake_class_dist = self._tensor_class_center_distance(self.X_hat, self.L)
        if self.use_class_conditioning:
            self.err_gate_bal = self.X.new_zeros(())
        else:
            self.err_gate_bal = self._gate_balance_loss(self.gate_weights)
        self.err_router = self._router_supervision_loss(self.gate_weights, self.L)
        if self.Z_mode_seek is not None and self.H_hat_mode_seek is not None:
            self.err_mode_seek = self._mode_seeking_loss(self.Z, self.Z_mode_seek, self.H_hat, self.H_hat_mode_seek)
        else:
            self.err_mode_seek = self.X.new_zeros(())
        self.err_class_center, self.err_class_distance = self._class_structure_loss(self.X, self.X_hat, labels=self.L)
        use_compact = bool(getattr(self.opt, "use_compact_loss", False))
        if use_compact:
            self.err_g_V1 = self.X.new_zeros(())
            self.err_g_V2 = self.X.new_zeros(())
            self.err_ms = self.X.new_zeros(())
            self.err_freq = self.X.new_zeros(())
            self.err_trend = self.X.new_zeros(())
            self.err_seasonal_freq = self.X.new_zeros(())
            self.err_seasonal_acf = self.X.new_zeros(())
            self.err_swd = self.X.new_zeros(())
            self.err_roll = self.X.new_zeros(())
            self.err_roll_level_std = self.X.new_zeros(())
            self.err_roll_diff_mean = self.X.new_zeros(())
            self.err_roll_diff_std = self.X.new_zeros(())
            self.err_roll_acf = self.X.new_zeros(())
            self.err_compact = self._compact_stat_loss(self.X, self.X_hat, labels=self.L)
            if not torch.isfinite(self.err_compact):
                self.err_compact = self.X.new_zeros(())
            rollout_lambda = 0.0
        else:
            self.err_compact = self.X.new_zeros(())
            self.err_compact_level = self.X.new_zeros(())
            self.err_compact_diff = self.X.new_zeros(())
            self.err_compact_acf = self.X.new_zeros(())
            self.err_g_V1, self.err_g_V2 = self._moment_loss(self.X, self.X_hat, labels=self.L)
            self.err_ms = self._multiscale_loss(self.X, self.X_hat, labels=self.L)
            self.err_freq = self._frequency_loss(self.X, self.X_hat, labels=self.L)
            self.err_trend = self._trend_loss(self.X, self.X_hat, labels=self.L)
            self.err_seasonal_freq = self._seasonal_frequency_loss(self.X, self.X_hat, labels=self.L)
            self.err_seasonal_acf = self._seasonal_acf_loss(self.X, self.X_hat, labels=self.L)
            self.err_swd = self._sliced_wasserstein_loss(
                self.X, self.X_hat, num_projections=int(getattr(self.opt, "swd_num_proj", 64)), labels=self.L
            )
            rollout_lambda = float(getattr(self.opt, "lambda_rollout", 0.0))
            if rollout_lambda > 0.0:
                roll_len = self._rollout_len(self.X.size(1))
                max_start = max(0, int(self.X.size(1)) - int(roll_len))
                start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
                real_seg = self.X[:, start : start + roll_len, :]
                fake_seg = self.X_hat[:, start : start + roll_len, :]
                self.err_roll = self._rollout_reg_loss(real_seg, fake_seg)
                if not torch.isfinite(self.err_roll):
                    self.err_roll = self.X.new_zeros(())
            else:
                self.err_roll = self.X.new_zeros(())
                self.err_roll_level_std = self.X.new_zeros(())
                self.err_roll_diff_mean = self.X.new_zeros(())
                self.err_roll_diff_std = self.X.new_zeros(())
                self.err_roll_acf = self.X.new_zeros(())

        div_margin = float(getattr(self.opt, "diversity_margin", 0.0))
        self.err_div = self.netg.diversity_loss(self.all_E_hat, margin=div_margin)
        adv_factor = self._adv_factor()
        cross_factor = self._cross_factor()
        self.err_g_adv_group = adv_factor * (self.err_g_U + self.err_g_U_e * self.opt.w_gamma)
        self.err_g_sup_group = self.opt.w_s * torch.sqrt(self.err_s + 1e-8) + self.opt.lambda_s * self.err_s
        self.err_g_div_group = cross_factor * (
            self.opt.lambda_cross_g * self.err_div
            + float(getattr(self.opt, "lambda_mode_seeking", 0.0)) * self.err_mode_seek
            + self.opt.lambda_gate_balance * self.err_gate_bal
            + float(getattr(self.opt, "lambda_router_supervision", 0.0)) * self.err_router
        )

        if use_compact:
            self.err_g_stat_group = (
                float(getattr(self.opt, "lambda_compact", 1.0)) * self.err_compact
                + float(getattr(self.opt, "lambda_class_center", 0.0)) * self.err_class_center
                + float(getattr(self.opt, "lambda_class_distance", 0.0)) * self.err_class_distance
            )
            self.err_g = (
                float(getattr(self.opt, "lambda_adv_group", 1.0)) * self.err_g_adv_group
                + float(getattr(self.opt, "lambda_sup_group", 1.0)) * self.err_g_sup_group
                + float(getattr(self.opt, "lambda_stat_group", 1.0)) * self.err_g_stat_group
                + float(getattr(self.opt, "lambda_div_group", 1.0)) * self.err_g_div_group
            )
        else:
            self.err_g_stat_group = (
                self.err_g_V1 * self.opt.w_g
                + self.err_g_V2 * self.opt.w_g
                + self.opt.lambda_ms * self.err_ms
                + self.opt.lambda_freq * self.err_freq
                + self.opt.lambda_trend * self.err_trend
                + float(getattr(self.opt, "lambda_seasonal_freq", 0.0)) * self.err_seasonal_freq
                + float(getattr(self.opt, "lambda_seasonal_acf", 0.0)) * self.err_seasonal_acf
                + self.opt.lambda_swd * self.err_swd
                + float(getattr(self.opt, "lambda_class_center", 0.0)) * self.err_class_center
                + float(getattr(self.opt, "lambda_class_distance", 0.0)) * self.err_class_distance
                + rollout_lambda * self.err_roll
            )
            self.err_g = (
                float(getattr(self.opt, "lambda_adv_group", 1.0)) * self.err_g_adv_group
                + float(getattr(self.opt, "lambda_sup_group", 1.0)) * self.err_g_sup_group
                + float(getattr(self.opt, "lambda_stat_group", 1.0)) * self.err_g_stat_group
                + float(getattr(self.opt, "lambda_div_group", 1.0)) * self.err_g_div_group
            )
        self.err_g.backward(retain_graph=True)

    def backward_s(self):
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)

    def backward_d(self):
        self.err_d_real = self.l_bce(self.Y_real, self._label(self.Y_real, is_real=True))
        self.err_d_fake = self.l_bce(self.Y_fake, self._label(self.Y_fake, is_real=False))
        self.err_d_fake_e = self.l_bce(self.Y_fake_e, self._label(self.Y_fake_e, is_real=False))
        adv_factor = self._adv_factor()
        self.err_d = self.err_d_real + adv_factor * (self.err_d_fake + self.err_d_fake_e * self.opt.w_gamma)
        if float(self.err_d.detach().cpu()) > 0.15:
            self.err_d.backward(retain_graph=True)

    def optimize_params_er(self):
        self.forward_er()
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self._clip_grad([self.nete, self.netr])
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self):
        self.forward_er()
        self.forward_s()
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_()
        self._clip_grad([self.nete, self.netr])
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        self.forward_e()
        self.forward_s()
        self.optimizer_s.zero_grad()
        self.backward_s()
        self._clip_grad([self.nets])
        self.optimizer_s.step()

    def optimize_params_g(self):
        self.forward_e()
        self.forward_s()
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g()
        self._clip_grad([self.netg, self.nets])
        self.optimizer_g.step()
        self.optimizer_s.step()

    def optimize_params_d(self):
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        self.forward_dg()

        self.optimizer_d.zero_grad()
        self.backward_d()
        self._clip_grad([self.netd])
        self.optimizer_d.step()
