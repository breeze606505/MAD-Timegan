"""Shared utility helpers for MAD-TimeGAN."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def to_numpy_batch(data, n=None):
    arr = np.asarray(data)
    if arr.dtype == object:
        arr = np.asarray([np.asarray(x) for x in data], dtype=np.float32)
    arr = arr.astype(np.float32)
    if n is not None:
        arr = arr[: int(n)]
    return arr


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Split original and generated data into train and test partitions."""
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Return per-sample length and maximum sequence length."""
    time = []
    max_seq_len = 0
    for sample in data:
        seq_len = len(sample[:, 0])
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)
    return time, max_seq_len


def random_generator(batch_size, z_dim, t_mb, max_seq_len):
    """Generate zero-padded latent noise for variable-length batches."""
    z_mb = []
    for i in range(int(batch_size)):
        temp = np.zeros([int(max_seq_len), int(z_dim)], dtype=np.float32)
        seq_len = int(t_mb[i])
        temp[:seq_len, :] = np.random.uniform(0.0, 1.0, [seq_len, int(z_dim)])
        z_mb.append(temp)
    return z_mb


def NormMinMax(data):
    """Normalize a batch of sequences and return min/max for renormalization."""
    arr = to_numpy_batch(data)
    min_val = np.min(np.min(arr, axis=0), axis=0)
    shifted = arr - min_val
    max_val = np.max(np.max(shifted, axis=0), axis=0)
    norm_data = shifted / (max_val + 1e-7)
    return norm_data.astype(np.float32), min_val.astype(np.float32), max_val.astype(np.float32)


def _flatten_samples(data):
    return [np.asarray(sample).reshape(-1) for sample in data]


def _balanced_label_subset(data, labels, max_items, rng):
    data_np = to_numpy_batch(data)
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(data_np) != len(labels_np):
        raise ValueError("Data and labels must have the same length.")
    if max_items is None or int(max_items) <= 0 or len(data_np) <= int(max_items):
        return data_np, labels_np

    unique_labels = sorted(np.unique(labels_np).tolist())
    per_class_cap = max(1, int(np.ceil(float(max_items) / float(max(1, len(unique_labels))))))
    selected = []
    for label in unique_labels:
        label_indices = np.where(labels_np == label)[0]
        if len(label_indices) > per_class_cap:
            label_indices = rng.choice(label_indices, size=per_class_cap, replace=False)
        selected.extend(np.asarray(label_indices, dtype=np.int64).tolist())

    if len(selected) > int(max_items):
        selected = rng.choice(np.asarray(selected, dtype=np.int64), size=int(max_items), replace=False).tolist()

    selected = np.asarray(sorted(selected), dtype=np.int64)
    return data_np[selected], labels_np[selected]


def visualization(ori_data, generated_data, analysis="pca", save_path=None, sample_num=500):
    """Visualize the overlap between real and synthetic samples."""
    ori_np = to_numpy_batch(ori_data)
    gen_np = to_numpy_batch(generated_data)
    if len(ori_np) == 0 or len(gen_np) == 0:
        raise ValueError("Need at least one real sample and one generated sample for visualization.")

    rng = np.random.default_rng(42)
    if len(ori_np) > sample_num:
        ori_np = ori_np[rng.choice(len(ori_np), sample_num, replace=False)]
    if len(gen_np) > sample_num:
        gen_np = gen_np[rng.choice(len(gen_np), sample_num, replace=False)]

    ori_flat = _flatten_samples(ori_np)
    gen_flat = _flatten_samples(gen_np)
    combined = np.concatenate([ori_flat, gen_flat], axis=0)

    analysis = str(analysis).lower()
    if analysis == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(combined)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    elif analysis == "tsne":
        total = len(combined)
        perplexity = max(1, min(40, total - 1, max(5, (total - 1) // 3)))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(combined)
    else:
        raise ValueError("analysis must be 'pca' or 'tsne'")

    num_real = len(ori_flat)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced[:num_real, 0], reduced[:num_real, 1], c="red", alpha=0.5, label="Real")
    plt.scatter(reduced[num_real:, 0], reduced[num_real:, 1], c="blue", alpha=0.5, label="Synthetic")
    plt.legend()
    plt.title(f"{analysis.upper()} Visualization of MAD-TimeGAN")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Visualization] Saved {analysis.upper()} plot to: {save_path}")
    else:
        plt.show()


def visualization_by_labels(
    ori_data,
    generated_data,
    ori_labels,
    generated_labels,
    analysis="pca",
    save_path=None,
    sample_num=500,
):
    """Visualize real vs synthetic samples with class labels overlaid."""
    rng = np.random.default_rng(42)
    ori_np, ori_labels_np = _balanced_label_subset(ori_data, ori_labels, sample_num, rng)
    gen_np, gen_labels_np = _balanced_label_subset(generated_data, generated_labels, sample_num, rng)
    if len(ori_np) == 0 or len(gen_np) == 0:
        raise ValueError("Need at least one real sample and one generated sample for visualization.")

    ori_flat = _flatten_samples(ori_np)
    gen_flat = _flatten_samples(gen_np)
    combined = np.concatenate([ori_flat, gen_flat], axis=0)

    analysis = str(analysis).lower()
    if analysis == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(combined)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    elif analysis == "tsne":
        total = len(combined)
        perplexity = max(1, min(40, total - 1, max(5, (total - 1) // 3)))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(combined)
    else:
        raise ValueError("analysis must be 'pca' or 'tsne'")

    num_real = len(ori_flat)
    real_reduced = reduced[:num_real]
    fake_reduced = reduced[num_real:]
    label_set = sorted(set(ori_labels_np.tolist()) | set(gen_labels_np.tolist()))
    cmap = plt.cm.get_cmap("tab10", max(1, len(label_set)))

    plt.figure(figsize=(8, 8))
    for idx, label in enumerate(label_set):
        color = cmap(idx)
        real_mask = ori_labels_np == label
        fake_mask = gen_labels_np == label
        if np.any(real_mask):
            plt.scatter(
                real_reduced[real_mask, 0],
                real_reduced[real_mask, 1],
                color=[color],
                alpha=0.55,
                marker="o",
                label=f"Real c{label}",
            )
        if np.any(fake_mask):
            plt.scatter(
                fake_reduced[fake_mask, 0],
                fake_reduced[fake_mask, 1],
                color=[color],
                alpha=0.85,
                marker="x",
                label=f"Synthetic c{label}",
            )

    plt.legend(ncol=2)
    plt.title(f"{analysis.upper()} Visualization by Label")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Visualization] Saved {analysis.upper()} label plot to: {save_path}")
    else:
        plt.show()


def save_label_visualizations(real_data, generated_data, real_labels, generated_labels, out_dir, sample_num=500):
    """Save overlay and per-class visualizations when labels are available."""
    if real_labels is None or generated_labels is None:
        return

    real_np = to_numpy_batch(real_data)
    gen_np = to_numpy_batch(generated_data)
    real_labels_np = np.asarray(real_labels, dtype=np.int64).reshape(-1)
    gen_labels_np = np.asarray(generated_labels, dtype=np.int64).reshape(-1)
    if len(real_np) != len(real_labels_np) or len(gen_np) != len(gen_labels_np):
        return

    os.makedirs(out_dir, exist_ok=True)
    visualization_by_labels(
        real_np,
        gen_np,
        real_labels_np,
        gen_labels_np,
        "pca",
        save_path=os.path.join(out_dir, "pca_visualization_by_label.png"),
        sample_num=sample_num,
    )
    visualization_by_labels(
        real_np,
        gen_np,
        real_labels_np,
        gen_labels_np,
        "tsne",
        save_path=os.path.join(out_dir, "tsne_visualization_by_label.png"),
        sample_num=sample_num,
    )

    label_set = sorted(set(real_labels_np.tolist()) | set(gen_labels_np.tolist()))
    for label in label_set:
        real_mask = real_labels_np == label
        fake_mask = gen_labels_np == label
        if int(np.sum(real_mask)) < 2 or int(np.sum(fake_mask)) < 2:
            continue
        class_dir = os.path.join(out_dir, f"class_{label}")
        visualization(
            real_np[real_mask],
            gen_np[fake_mask],
            "pca",
            save_path=os.path.join(class_dir, "pca_visualization.png"),
            sample_num=sample_num,
        )
        visualization(
            real_np[real_mask],
            gen_np[fake_mask],
            "tsne",
            save_path=os.path.join(class_dir, "tsne_visualization.png"),
            sample_num=sample_num,
        )


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def save_best_match_plots(real_data, fake_data, out_dir, top_k=3, max_reference_samples=2000):
    """Save nearest-match line plots for a few generated samples."""
    real_np = to_numpy_batch(real_data)
    fake_np = to_numpy_batch(fake_data)
    if len(real_np) == 0 or len(fake_np) == 0:
        return

    max_reference_samples = int(max_reference_samples)
    if max_reference_samples > 0 and len(real_np) > max_reference_samples:
        rng = np.random.default_rng(42)
        ref_idx = rng.choice(len(real_np), size=max_reference_samples, replace=False)
        real_np = real_np[ref_idx]

    pairs = []
    for g_idx, generated in enumerate(fake_np):
        rmse_errors = [_rmse(generated, real) for real in real_np]
        mae_errors = [_mae(generated, real) for real in real_np]
        best_idx = int(np.argmin(rmse_errors))
        pairs.append((g_idx, best_idx, rmse_errors[best_idx], mae_errors[best_idx]))

    pairs = sorted(pairs, key=lambda item: item[2])[: max(1, int(top_k))]
    feat_dim = real_np.shape[2]
    os.makedirs(out_dir, exist_ok=True)

    for rank, (g_idx, r_idx, rmse_err, mae_err) in enumerate(pairs, start=1):
        fig, axes = plt.subplots(feat_dim, 1, figsize=(10, 3.2 * feat_dim))
        if feat_dim == 1:
            axes = [axes]
        fig.suptitle(f"Best Match {rank} | RMSE={rmse_err:.6f} | MAE={mae_err:.6f}")

        for feature_idx in range(feat_dim):
            axes[feature_idx].plot(real_np[r_idx][:, feature_idx], label="Real", color="blue", linewidth=2)
            axes[feature_idx].plot(fake_np[g_idx][:, feature_idx], label="Generated", color="red", linestyle="--", linewidth=2)
            axes[feature_idx].set_title(f"Feature {feature_idx + 1}")
            axes[feature_idx].legend()
            axes[feature_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_best_{rank}.png"), dpi=300)
        plt.close()
