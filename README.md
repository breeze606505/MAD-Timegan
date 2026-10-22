

## Project Layout

- `train.py`: general training entrypoint
- `visualize.py`: regenerate PCA/TSNE and best-match plots from saved arrays
- `options.py`: CLI arguments
- `lib/`: model, training, data, utilities


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


python train.py --data_name Windpower_all_2dim --seq_len 512 --sample_num 256 --lambda_ms 0.5 --lambda_freq 0.5 --lambda_trend 0.5 --lambda_seasonal_freq 2.0 --lambda_seasonal_acf 5.0 --seasonal_periods auto
```
