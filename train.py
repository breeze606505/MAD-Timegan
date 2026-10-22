from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import numpy as np

from options import Options
from lib.data import load_data_with_labels
from lib.timegan import TimeGAN
from lib.utils import save_best_match_plots, save_label_visualizations, to_numpy_batch, visualization

warnings.filterwarnings("ignore")


def train():
    opt = Options().parse()
    ori_data, ori_labels = load_data_with_labels(opt)

    if len(ori_data) > 0:
        opt.z_dim = int(ori_data[0].shape[-1])

    model = TimeGAN(opt, ori_data, ori_labels=ori_labels)
    model.train()

    real_data = model.get_real_data(normalized=False)
    real_labels = model.get_real_labels()
    generated_payload = getattr(model, "generated_data", None)
    generated_label_payload = getattr(model, "generated_labels", None)
    if generated_payload is None:
        generated_payload, generated_label_payload = model.generation(opt.generate_num, return_labels=True)
    generated_data = generated_payload
    generated_np = to_numpy_batch(generated_data)

    out_dir = os.path.join(opt.outf, opt.name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "real.npy"), real_data)
    np.save(os.path.join(out_dir, "generated.npy"), generated_np)
    if real_labels is not None:
        np.save(os.path.join(out_dir, "real_labels.npy"), real_labels)
    if generated_label_payload is not None and len(generated_label_payload) > 0:
        np.save(os.path.join(out_dir, "generated_labels.npy"), generated_label_payload)

    pca_path = os.path.join(out_dir, "pca_visualization.png")
    tsne_path = os.path.join(out_dir, "tsne_visualization.png")
    visualization(real_data, generated_np, "pca", save_path=pca_path)
    visualization(real_data, generated_np, "tsne", save_path=tsne_path)
    save_best_match_plots(real_data, generated_np, out_dir, top_k=3)
    if real_labels is not None and generated_label_payload is not None and len(generated_label_payload) > 0:
        save_label_visualizations(real_data, generated_np, real_labels, generated_label_payload, out_dir)

    print(generated_np.shape)
    print("Finish synthetic data generation")
    print(f"Saved training outputs to: {out_dir}")


if __name__ == "__main__":
    train()
