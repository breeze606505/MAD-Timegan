"""Standalone visualization for saved real/generated arrays."""

import argparse
import os

import numpy as np

from lib.utils import save_best_match_plots, save_label_visualizations, visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True, help="path to real.npy")
    parser.add_argument("--generated_path", type=str, required=True, help="path to generated.npy")
    parser.add_argument("--real_labels_path", type=str, default="", help="optional path to real_labels.npy")
    parser.add_argument("--generated_labels_path", type=str, default="", help="optional path to generated_labels.npy")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_reference_samples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    real = np.load(args.real_path)
    generated = np.load(args.generated_path)

    os.makedirs(args.out_dir, exist_ok=True)
    visualization(real, generated, "pca", save_path=os.path.join(args.out_dir, "pca_visualization.png"), sample_num=args.sample_num)
    visualization(real, generated, "tsne", save_path=os.path.join(args.out_dir, "tsne_visualization.png"), sample_num=args.sample_num)
    save_best_match_plots(
        real,
        generated,
        args.out_dir,
        top_k=args.top_k,
        max_reference_samples=args.max_reference_samples,
    )

    real_labels_path = args.real_labels_path or os.path.join(os.path.dirname(args.real_path), "real_labels.npy")
    generated_labels_path = args.generated_labels_path or os.path.join(
        os.path.dirname(args.generated_path), "generated_labels.npy"
    )
    if os.path.exists(real_labels_path) and os.path.exists(generated_labels_path):
        real_labels = np.load(real_labels_path)
        generated_labels = np.load(generated_labels_path)
        save_label_visualizations(
            real,
            generated,
            real_labels,
            generated_labels,
            args.out_dir,
            sample_num=args.sample_num,
        )
    print(f"Saved visualization outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
