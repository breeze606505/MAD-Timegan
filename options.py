"""Command line options for MAD-TimeGAN."""

import argparse
import os

import torch


SUPPORTED_DATASETS = [
    "sine",
    "stock",
    "energy",
    "Windpower_damage",
    "Windpower_health",
    "Windpower_all",
    "Windpower_all_2dim",
    "Windpower_all_6dim",
    "Windpower_damage-2dim",
    "Windpower_health-2dim",
    "Windpower_damage-6dim",
    "Windpower_health-6dim",
    "custom_csv",
    "custom_mat",
]


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument("--data_name", choices=SUPPORTED_DATASETS, default="stock", type=str)
        self.parser.add_argument("--data_path", type=str, default="", help="path for custom_csv or custom_mat")
        self.parser.add_argument("--mat_vars", type=str, default="", help="comma-separated variables for custom_mat")
        self.parser.add_argument("--num_classes", type=int, default=0, help="number of class labels for conditional training")
        self.parser.add_argument("--z_dim", default=6, type=int, help="noise feature dimension")
        self.parser.add_argument("--feature_dim", default=None, type=int, help="real data feature dimension")
        self.parser.add_argument("--seq_len", default=24, type=int, help="sequence length")
        self.parser.add_argument("--hidden_dim", default=24, type=int, help="hidden state dimension")
        self.parser.add_argument("--num_layer", default=3, type=int, help="number of GRU layers")
        self.parser.add_argument("--num_layers", default=None, type=int, help="alias of num_layer")
        self.parser.add_argument("--iteration", default=50000, type=int, help="training iterations per stage")
        self.parser.add_argument("--batch_size", default=128, type=int, help="mini-batch size")
        self.parser.add_argument("--metric_iteration", default=10, type=int)
        self.parser.add_argument("--sample_num", type=int, default=None, help="number of samples loaded from dataset")
        self.parser.add_argument("--window_stride", type=int, default=1, help="sliding-window stride")
        self.parser.add_argument("--generate_num", type=int, default=500, help="number of synthetic samples")

        self.parser.add_argument("--workers", type=int, default=8)
        self.parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | gpu")
        self.parser.add_argument("--gpu_ids", type=str, default="0")
        self.parser.add_argument("--ngpu", type=int, default=1)
        self.parser.add_argument("--model", type=str, default="MAD-TimeGAN")
        self.parser.add_argument("--outf", default="./output", help="output directory")
        self.parser.add_argument("--name", type=str, default="experiment_name", help="experiment name")
        self.parser.add_argument("--manualseed", default=42, type=int)

        self.parser.add_argument("--print_freq", type=int, default=1000)
        self.parser.add_argument("--load_weights", action="store_true")
        self.parser.add_argument("--resume", default="", help="path to checkpoints")

        self.parser.add_argument("--beta1", type=float, default=0.9)
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument("--e_lr", type=float, default=None, help="encoder lr (defaults to --lr)")
        self.parser.add_argument("--r_lr", type=float, default=None, help="recovery lr (defaults to --lr)")
        self.parser.add_argument("--g_lr", type=float, default=None, help="generator lr (defaults to --lr)")
        self.parser.add_argument("--d_lr", type=float, default=None, help="discriminator lr (defaults to --lr)")
        self.parser.add_argument("--s_lr", type=float, default=None, help="supervisor lr (defaults to --lr)")
        self.parser.add_argument("--grad_clip", type=float, default=1.0, help="global grad-norm clip; <=0 disables")

        self.parser.add_argument("--w_gamma", type=float, default=1.0)
        self.parser.add_argument("--w_es", type=float, default=0.1)
        self.parser.add_argument("--w_e0", type=float, default=10.0)
        self.parser.add_argument("--w_g", type=float, default=100.0)
        self.parser.add_argument("--w_s", type=float, default=1.0)
        self.parser.add_argument("--lambda_adv_group", type=float, default=1.0, help="overall adversarial group weight")
        self.parser.add_argument("--lambda_sup_group", type=float, default=1.0, help="overall supervision group weight")
        self.parser.add_argument("--lambda_stat_group", type=float, default=1.0, help="overall statistical alignment group weight")
        self.parser.add_argument("--lambda_div_group", type=float, default=1.0, help="overall diversity/gating group weight")
        self.parser.add_argument("--lambda_ms", type=float, default=0.0)
        self.parser.add_argument("--lambda_freq", type=float, default=0.0)
        self.parser.add_argument("--lambda_trend", type=float, default=0.0)
        self.parser.add_argument(
            "--lambda_seasonal_freq",
            type=float,
            default=0.0,
            help="dominant-frequency alignment weight for periodic signals",
        )
        self.parser.add_argument(
            "--lambda_seasonal_acf",
            type=float,
            default=0.0,
            help="period-lag autocorrelation alignment weight for periodic signals",
        )
        self.parser.add_argument("--lambda_swd", type=float, default=0.0, help="sliced wasserstein loss weight")
        self.parser.add_argument(
            "--lambda_mode_seeking",
            type=float,
            default=0.0,
            help="single-generator mode seeking regularization weight",
        )
        self.parser.add_argument(
            "--mode_seek_eps",
            type=float,
            default=1e-4,
            help="stability epsilon for mode seeking regularization",
        )
        self.parser.add_argument(
            "--lambda_class_center",
            type=float,
            default=1.0,
            help="class-conditional center alignment weight; only active when labels exist",
        )
        self.parser.add_argument(
            "--lambda_class_distance",
            type=float,
            default=1.0,
            help="class-conditional inter-class distance preservation weight; only active when labels exist",
        )
        self.parser.add_argument("--swd_num_proj", type=int, default=64, help="number of random projections for SWD")
        self.parser.add_argument("--lambda_s", type=float, default=0.0)
        self.parser.add_argument("--lambda_gate_balance", type=float, default=0.0)
        self.parser.add_argument("--lambda_recon_l1", type=float, default=0.0)
        self.parser.add_argument("--ms_scales", type=str, default="2,4,8", help="comma-separated scales for multi-scale loss")
        self.parser.add_argument(
            "--seasonal_periods",
            type=str,
            default="auto",
            help="comma-separated dominant periods to enforce, or 'auto'",
        )
        self.parser.add_argument("--seasonal_topk", type=int, default=3, help="top dominant periods to keep in auto mode")
        self.parser.add_argument("--seasonal_min_period", type=int, default=8, help="minimum allowed seasonal period")
        self.parser.add_argument(
            "--seasonal_max_period",
            type=int,
            default=0,
            help="maximum allowed seasonal period; <=0 means seq_len//2",
        )
        self.parser.add_argument("--seasonal_harmonics", type=int, default=2, help="how many harmonics/lag multiples to enforce")

        self.parser.add_argument("--lambda_rollout", type=float, default=0.0, help="overall rollout regularization weight")
        self.parser.add_argument("--rollout_start", type=int, default=6, help="initial rollout length")
        self.parser.add_argument("--rollout_ramp_steps", type=int, default=2000, help="steps to ramp rollout length")
        self.parser.add_argument("--rollout_cap", type=int, default=0, help="max rollout length; <=0 means full seq_len")
        self.parser.add_argument("--rollout_acf_lags", type=str, default="1,2,4,8", help="comma-separated lags for rollout ACF loss")
        self.parser.add_argument("--w_rollout_level_std", type=float, default=0.35)
        self.parser.add_argument("--w_rollout_diff_mean", type=float, default=0.15)
        self.parser.add_argument("--w_rollout_diff_std", type=float, default=0.30)
        self.parser.add_argument("--w_rollout_acf", type=float, default=0.20)

        self.parser.add_argument(
            "--use_compact_loss",
            action="store_true",
            default=False,
            help="replace overlapping generator regularizers with compact statistical loss",
        )
        self.parser.add_argument("--lambda_compact", type=float, default=0.0, help="overall compact statistical loss weight")
        self.parser.add_argument("--compact_acf_lags", type=str, default="1,2,4,8", help="comma-separated lags for compact ACF loss")
        self.parser.add_argument("--w_compact_level", type=float, default=0.35)
        self.parser.add_argument("--w_compact_diff", type=float, default=0.35)
        self.parser.add_argument("--w_compact_acf", type=float, default=0.30)
        self.parser.add_argument("--diversity_margin", type=float, default=0.0)
        self.parser.add_argument("--use_sigmoid_output", action="store_true", default=True)
        self.parser.add_argument("--no_use_sigmoid_output", action="store_false", dest="use_sigmoid_output")
        self.parser.add_argument("--adv_warmup_steps", type=int, default=0, help="warmup steps for adversarial loss in stage-3")
        self.parser.add_argument("--cross_warmup_steps", type=int, default=0, help="warmup steps for cross-generator regularizers")

        self.parser.add_argument("--use_metric_earlystop", action="store_true", default=False)
        self.parser.add_argument("--es_eval_interval", type=int, default=200, help="stage-3 evaluation interval")
        self.parser.add_argument("--es_patience", type=int, default=8, help="early-stop patience by eval rounds")
        self.parser.add_argument("--es_min_delta", type=float, default=1e-3, help="minimum improvement threshold")
        self.parser.add_argument("--es_eval_samples", type=int, default=256, help="samples used in each metric evaluation")
        self.parser.add_argument("--es_warmup_steps", type=int, default=1000, help="ignore early-stop before this stage-3 step")
        self.parser.add_argument("--es_ema_beta", type=float, default=0.6, help="EMA smoothing factor for early-stop score")
        self.parser.add_argument(
            "--es_metric",
            type=str,
            default="nn_combo",
            choices=["nn_combo", "nn_rmse", "nn_mae", "paired_rmse", "paired_mae"],
            help="metric used for early-stop; lower is better",
        )
        self.parser.add_argument("--es_mae_weight", type=float, default=0.5, help="weight for nn_mae in nn_combo score")

        self.parser.add_argument("--generator_arch", type=str, default="transgru", choices=["gru", "transgru"])
        self.parser.add_argument("--num_generators", type=int, default=3)
        self.parser.add_argument("--mode_selection", type=str, default="hybrid", choices=["disc", "soft", "hybrid"])
        self.parser.add_argument("--use_class_conditioning", action="store_true", default=True)
        self.parser.add_argument("--no_use_class_conditioning", action="store_false", dest="use_class_conditioning")
        self.parser.add_argument("--classwise_stat_loss", action="store_true", default=True)
        self.parser.add_argument("--no_classwise_stat_loss", action="store_false", dest="classwise_stat_loss")
        self.parser.add_argument("--train_hard_routing", action="store_true", default=True)
        self.parser.add_argument("--no_train_hard_routing", action="store_false", dest="train_hard_routing")
        self.parser.add_argument(
            "--class_generator_match",
            action="store_true",
            default=True,
            help="when num_generators == num_classes, route class k to generator k",
        )
        self.parser.add_argument(
            "--no_class_generator_match",
            action="store_false",
            dest="class_generator_match",
        )
        self.parser.add_argument(
            "--condition_sampling",
            type=str,
            default="empirical",
            choices=["empirical", "balanced"],
            help="label sampling strategy during unconditional generation",
        )
        self.parser.add_argument("--lambda_cross_g", type=float, default=0.1, help="diversity regularization")
        self.parser.add_argument(
            "--lambda_router_supervision",
            type=float,
            default=1.0,
            help="router supervision weight when class labels align with generators",
        )
        self.parser.add_argument("--nhead", type=int, default=4, help="transformer attention heads")
        self.parser.add_argument("--transformer_layers", type=int, default=1)
        self.parser.add_argument("--ffn_dim", type=int, default=256)
        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--gen_residual", action="store_true", default=True)
        self.parser.add_argument("--no_gen_residual", action="store_false", dest="gen_residual")
        self.parser.add_argument("--gen_layernorm", action="store_true", default=True)
        self.parser.add_argument("--no_gen_layernorm", action="store_false", dest="gen_layernorm")

        self.parser.add_argument("--prune_amount", type=float, default=0.0, help="0~1")
        self.parser.add_argument("--prune_mode", type=str, default="hybrid", choices=["weights", "generators", "hybrid"])
        self.parser.add_argument("--post_prune_finetune_iters", type=int, default=0)

        self.parser.add_argument("--stage1_iters", type=int, default=None, help="stage-1 ER iterations")
        self.parser.add_argument("--stage2_iters", type=int, default=None, help="stage-2 S iterations")
        self.parser.add_argument("--stage3_iters", type=int, default=None, help="stage-3 joint iterations")
        self.parser.add_argument("--joint_g_steps", type=int, default=2, help="G/S updates per stage-3 step")

        self.parser.add_argument("--disc_use_logits", action="store_true", default=True)
        self.parser.add_argument("--no_disc_use_logits", action="store_false", dest="disc_use_logits")
        self.parser.add_argument("--disc_spectral_norm", action="store_true", default=False)
        self.parser.add_argument("--label_smooth_real", type=float, default=1.0)
        self.parser.add_argument("--label_smooth_fake", type=float, default=0.0)
        self.parser.add_argument("--instance_noise_std", type=float, default=0.0)

        self.parser.add_argument("--reproduce_ress", action="store_true", help="use RESS-like TransGRU configuration")

        self.isTrain = True
        self.opt = None

    @staticmethod
    def _parse_int_tokens(text, min_value):
        values = []
        for token in str(text).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            if value >= min_value:
                values.append(value)
        return sorted(set(values))

    def _parse_gpu_ids(self):
        gpu_ids = []
        for token in str(self.opt.gpu_ids).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx >= 0:
                gpu_ids.append(idx)
        self.opt.gpu_ids = gpu_ids

    def _resolve_device(self):
        requested = str(self.opt.device).lower()
        wants_cuda = requested in {"auto", "gpu", "cuda"}
        has_cuda = torch.cuda.is_available() and bool(self.opt.gpu_ids)

        if wants_cuda and has_cuda:
            self.opt.device = "cuda"
            torch.cuda.set_device(self.opt.gpu_ids[0])
            return

        if requested in {"gpu", "cuda"} and not has_cuda:
            print("[Options] CUDA requested but unavailable; falling back to CPU.")

        self.opt.device = "cpu"
        self.opt.gpu_ids = []

    def _validate_dataset_args(self):
        if self.opt.data_name in {"custom_csv", "custom_mat"} and not self.opt.data_path:
            raise ValueError(f"--data_path is required when data_name={self.opt.data_name}")
        if self.opt.data_name == "custom_mat" and not self.opt.mat_vars:
            raise ValueError("--mat_vars is required when data_name=custom_mat")

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        if self.opt.num_layers is not None:
            self.opt.num_layer = int(self.opt.num_layers)

        self._parse_gpu_ids()
        self._resolve_device()
        self._validate_dataset_args()

        if self.opt.reproduce_ress:
            self.opt.generator_arch = "transgru"
            self.opt.num_generators = max(2, self.opt.num_generators)
            self.opt.nhead = 4 if self.opt.data_name in {"stock", "energy"} else 8
            self.opt.hidden_dim = max(self.opt.hidden_dim, 128)
            self.opt.batch_size = min(self.opt.batch_size, 32)
            self.opt.beta1 = 0.5
            self.opt.grad_clip = max(1.0, self.opt.grad_clip)
            self.opt.disc_use_logits = True
            self.opt.disc_spectral_norm = True
            self.opt.lambda_ms = max(self.opt.lambda_ms, 1.0)
            self.opt.lambda_freq = max(self.opt.lambda_freq, 0.5)
            self.opt.lambda_trend = max(self.opt.lambda_trend, 0.5)
            self.opt.lambda_gate_balance = max(self.opt.lambda_gate_balance, 0.05)
            warmup_floor = max(50, self.opt.iteration // 10)
            self.opt.adv_warmup_steps = max(self.opt.adv_warmup_steps, warmup_floor)
            self.opt.cross_warmup_steps = max(self.opt.cross_warmup_steps, warmup_floor)

        if self.opt.data_name == "energy" and self.opt.sample_num is None:
            self.opt.sample_num = 256

        if self.opt.name == "experiment_name":
            if self.opt.data_name in {"custom_csv", "custom_mat"}:
                data_tag = os.path.splitext(os.path.basename(self.opt.data_path))[0]
            else:
                data_tag = self.opt.data_name
            self.opt.name = f"{self.opt.model}/{data_tag}"

        self.opt.ms_scales = self._parse_int_tokens(self.opt.ms_scales, min_value=2) or [2, 4, 8]
        self.opt.rollout_acf_lags = self._parse_int_tokens(self.opt.rollout_acf_lags, min_value=1) or [1, 2, 4, 8]
        self.opt.compact_acf_lags = self._parse_int_tokens(self.opt.compact_acf_lags, min_value=1) or [1, 2, 4, 8]
        if self.opt.use_compact_loss and self.opt.lambda_compact <= 0.0:
            self.opt.lambda_compact = 1.0

        if self.opt.stage1_iters is None:
            self.opt.stage1_iters = int(self.opt.iteration)
        if self.opt.stage2_iters is None:
            self.opt.stage2_iters = int(self.opt.iteration)
        if self.opt.stage3_iters is None:
            self.opt.stage3_iters = int(self.opt.iteration)
        self.opt.es_eval_interval = max(1, int(self.opt.es_eval_interval))
        self.opt.es_patience = max(1, int(self.opt.es_patience))
        self.opt.es_eval_samples = max(8, int(self.opt.es_eval_samples))
        self.opt.es_warmup_steps = max(0, int(self.opt.es_warmup_steps))
        self.opt.es_ema_beta = float(min(0.99, max(0.0, self.opt.es_ema_beta)))
        self.opt.es_mae_weight = float(max(0.0, self.opt.es_mae_weight))

        for name in ["e_lr", "r_lr", "g_lr", "d_lr", "s_lr"]:
            if getattr(self.opt, name) is None:
                setattr(self.opt, name, float(self.opt.lr))

        expr_dir = os.path.join(self.opt.outf, self.opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        args = vars(self.opt)
        with open(os.path.join(expr_dir, "opt.txt"), "wt", encoding="utf-8") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for key, value in sorted(args.items()):
                opt_file.write(f"{key}: {value}\n")
            opt_file.write("-------------- End ----------------\n")

        return self.opt
