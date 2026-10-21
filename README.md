# MAD-TIMEGAN

Cleaned and generalized TimeGAN project for multivariate time-series generation and augmentation.

## What Changed

- Removed one-off launcher scripts from the project root.
- Kept the general training entrypoint: `train.py`.
- Kept standalone visualization via `visualize.py`.
- Kept reusable experiment scripts under `experiments/`.
- Fixed real/fake scale mismatches in training evaluation and saved visualizations.
- Fixed device fallback so CUDA requests degrade to CPU cleanly.
- Fixed variable-length noise padding in `random_generator`.
- Removed unsupported dataset options and added generic `custom_csv` / `custom_mat` loaders.
- Added `requirements.txt` and `.gitignore`.

## Project Layout

- `train.py`: general training entrypoint
- `visualize.py`: regenerate PCA/TSNE and best-match plots from saved arrays
- `options.py`: CLI arguments
- `lib/`: model, training, data, utilities
- `experiments/compare_augmentors.py`: compare MAD-TimeGAN vs baseline GAN augmentors
- `experiments/full_paper_experiments.py`: multi-dataset downstream benchmark
- `experiments/auto_tune_timegan.py`: random/grid hyperparameter search

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python train.py --data_name energy --device auto --reproduce_ress --num_generators 4 --mode_selection hybrid
```

For merged class datasets such as `Windpower_all_2dim` and `Windpower_all_6dim`, the training entrypoint now keeps class labels and enables conditional generation plus hard routing by default.

Example:

```bash
python train.py --data_name Windpower_all_2dim --device auto --num_generators 2 --mode_selection hybrid --lambda_gate_balance 0.0 --condition_sampling balanced
```

Outputs are saved under `output/<experiment_name>/`:

- `real.npy`
- `generated.npy`
- `real_labels.npy`
- `generated_labels.npy`
- `pca_visualization.png`
- `tsne_visualization.png`
- `pca_visualization_by_label.png`
- `tsne_visualization_by_label.png`
- `class_<label>/pca_visualization.png`
- `class_<label>/tsne_visualization.png`
- `comparison_best_*.png`
- `train/diagnostics/train_diagnostics.csv`
- `train/diagnostics/train_diagnostics.json`
- `train/weights/*.pth`

## Custom Data

Custom CSV:

```bash
python train.py --data_name custom_csv --data_path D:/mydata/my_series.csv --seq_len 48 --device auto
```

Custom MAT:

```bash
python train.py --data_name custom_mat --data_path D:/mydata/my_series.mat --mat_vars signal1,signal2 --seq_len 48 --device auto
```

## Visualization Only

```bash
python visualize.py --real_path output/MAD-TimeGAN/energy/real.npy --generated_path output/MAD-TimeGAN/energy/generated.npy --out_dir output/MAD-TimeGAN/energy/replots
```

## General Experiments

Compare augmentors:

```bash
python experiments/compare_augmentors.py --data_name energy --device cpu --real_train_cap 600 --aug_ratio 1.0 --tg_steps 200
```

Full benchmark:

```bash
python experiments/full_paper_experiments.py --datasets energy,stock,Windpower_all_2dim,Windpower_all_6dim --methods real_only,timegan_multi,timegan_pruned,wgan_gp,cgan --seeds 42,52,62 --device cpu --exp_name paper_full
```

Auto tuning:

```bash
python experiments/auto_tune_timegan.py --data_name energy --device cpu --outf ./output_tune --exp_name energy_tune --sample_num 2048 --batch_size 32 --eval_samples 512 --stage1_iters 200 --stage2_iters 200 --stage3_iters 400 --joint_g_steps 2 --search_mode random --trials 20
```

## Multi-Objective Training Notes

MAD-TimeGAN combines adversarial, supervised, reconstruction, and statistical alignment losses. In practice:

- Start with the stable core first: reconstruction + supervision + adversarial.
- Turn on extra regularizers one by one instead of all at once.
- Keep warmup for adversarial and cross-generator terms when adding many objectives.
- Prefer small auxiliary weights first, then scale up only if the target metric actually improves.
- Track each loss term separately and tune against downstream or sample-quality metrics, not raw total loss alone.
- `--print_freq` now also controls how often grouped training diagnostics are recorded.
- Use the new coarse group weights to tune at the right level first:
  - `--lambda_adv_group`
  - `--lambda_sup_group`
  - `--lambda_stat_group`
  - `--lambda_div_group`
- For labeled mixed datasets, separate the objectives conceptually:
  - within-class fit: `classwise_stat_loss`, `lambda_class_center`
  - between-class separation: `class_generator_match`, `lambda_router_supervision`, `lambda_class_distance`
- For known multi-class mixtures, keep these enabled:
  - `--use_class_conditioning`
  - `--classwise_stat_loss`
  - `--train_hard_routing`
  - `--class_generator_match`
- Recommended starting point for `Windpower_all_2dim`:

```bash
python train.py --data_name Windpower_all_2dim --device auto --num_generators 2 --mode_selection hybrid --condition_sampling balanced --lambda_gate_balance 0.0 --lambda_adv_group 0.7 --lambda_stat_group 2.0 --lambda_div_group 1.5 --lambda_cross_g 0.3 --lambda_router_supervision 3.0 --lambda_class_center 2.0 --lambda_class_distance 4.0
```

Useful diagnostic columns:

- `g_adv_group`, `g_sup_group`, `g_stat_group`, `g_div_group`: the four top-level objectives you actually balance.
- `g_router`, `g_class_center`, `g_class_distance`: the class-separation terms for mixed datasets.
- `g_seasonal_freq`, `g_seasonal_acf`: the period-specific losses for dominant periodic structure.
- `batch_fake_real_dist_ratio`: current fake inter-class distance divided by real inter-class distance. If it stays near `0`, the model is still collapsing classes together.
- `gate_mean_*` and `route_frac_*`: whether the router is actually using multiple generators.

For long sequences with obvious cycles, do not rely on mean/std losses alone. Turn on the periodic objectives and prefer training from weights with the same `seq_len`:

```bash
python train.py --data_name Windpower_all_2dim --seq_len 512 --sample_num 256 --lambda_ms 0.5 --lambda_freq 0.5 --lambda_trend 0.5 --lambda_seasonal_freq 2.0 --lambda_seasonal_acf 5.0 --seasonal_periods auto
```
