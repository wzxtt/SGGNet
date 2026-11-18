# SGGNet: Integrating Swin Transformer and Dynamic Snake Convolution for FAZ Segmentation in OCTA v1.0.0 (PLOS ONE revision)

This repository contains the official PyTorch implementation of SGGNet, a deep learning model for Foveal Avascular Zone (FAZ) segmentation in Optical Coherence Tomography Angiography (OCTA) images.

## Architecture

SGGNet employs a U-shaped encoder-decoder architecture that integrates Swin Transformer and Dynamic Snake Convolution:

- **Encoder**: Three SwinSnakeDual modules for hierarchical feature extraction with dual-branch downsampling
- **Decoder**: Three DSCG upsampling blocks with skip connection concatenation
- **Skip Connections**: Processed through Group Receptive Field Block (GRFB) for multi-scale feature fusion

### Key Components

| Module | Description |
|--------|-------------|
| **DSCG** | Dynamic Snake Convolution with GELU activation, combining standard conv with axis-aligned deformable convolutions |
| **SwinSnakeDual** | Dual-branch downsampling combining DSCG (CNN branch) and Swin Transformer V2 (attention branch) |
| **Swin Transformer V2** | Window-based multi-head attention with cosine similarity and log-spaced position bias |
| **GRFB** | Group Receptive Field Block with multi-branch parallel convolutions (3x3, 5x5, dilated-2, dilated-4) |

## Installation

```bash
# Create conda environment
conda create -n sggnet python=3.9
conda activate sggnet

# Install dependencies
pip install torch torchvision
pip install numpy scipy scikit-image pillow
pip install tensorboard pyyaml tqdm

# For evaluation metrics (optional)
pip install opencv-python
```

## Dataset Preparation

The model is designed for the OCTA-500 dataset. Organize the data as follows:

```
data/OCTA-500/
├── Projection/
│   ├── FAZ_Full/          # FULL projection layer (3 channels input)
│   ├── FAZ_ILM_OPL/       # ILM-OPL projection layer
│   └── FAZ_OPL_BM/        # OPL-BM projection layer
├── Label/
│   └── FAZ/               # Binary FAZ annotation masks
```

The three projection layers (FULL, ILM_OPL, OPL_BM) are stacked as 3-channel input to the network.

## Training

```bash
# Train with default settings
python train.py --data_root ./data/OCTA-500 --seed 42

# Train with custom parameters
python train.py \
    --data_root ./data/OCTA-500 \
    --seed 42 \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 2

# Train with all 5 seeds
for seed in 42 123 456 789 1024; do
    python train.py --data_root ./data/OCTA-500 --seed $seed
done
```

### Training Details

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: none
- **Loss**: BCE + Dice Loss
- **Input Size**: 304 x 304 (3M subset); 400 x 400 (6M subset)
- **Batch Size**: 2
- **Epochs**: 100
- **Data Augmentation**: Random horizontal flip, random vertical flip, and random 90-degree rotation, each applied with probability 0.5

### Dataset split

Experiments use the official OCTA-500 split: 3M has 140 training, 10 validation and 50 test images; 6M has 180, 20 and 100. Put the training and validation images in `data_root` and pass `--val_split 0.0667` for 3M or `--val_split 0.1` for 6M so that the validation count matches the paper; the test images are evaluated separately with `evaluate.py`.

## Data and reproduction files

The `reproduce/` directory contains the aggregate outputs behind the tables and the statistical analysis.

| File | Rows | Contents |
|------|------|----------|
| `supplementary_raw_results.csv` | 70 | Per-seed Dice, Jaccard and HD for the seven models in Table 2 (7 models x 5 seeds x 2 subsets). |
| `per_image_results.csv` | 1,050 | Per-image Dice, Jaccard and HD, averaged over the five seeds, for every test image and model (7 models x 150 test images). Input to the paired tests in Section 4.3 and to Supplementary Table S1. |
| `supplementary_pvalues.csv` | 36 | Paired t and exact Wilcoxon results, Holm-adjusted p-values, 95% confidence intervals and Cohen's d_z for every baseline and metric. |

`generate_supplementary_data.py` rebuilds the three files from raw model outputs, and `supplementary_fig_s1_loss_curves.py` regenerates Fig. S1.

Not included in this release: per-seed or per-image outputs for the five ablation variants in Table 5, per-image boundary metrics (HD95, ASSD, B-F1) behind Table 4, and the per-seed values behind Supplementary Tables S2 and S3. Those tables report mean values only.

## Evaluation

```bash
# Evaluate a single checkpoint
python evaluate.py \
    --data_root ./data/OCTA-500 \
    --checkpoint checkpoints/seed_42/best_model.pth

# Evaluate all seeds and get mean +/- std
python evaluate.py \
    --data_root ./data/OCTA-500 \
    --checkpoint checkpoints \
    --seeds 42 123 456 789 1024
```

### Metrics

| Metric | Description |
|--------|-------------|
| Dice | Dice Similarity Coefficient |
| Jaccard | Jaccard Index (IoU) |
| HD | Hausdorff Distance |
| HD95 | 95th percentile Hausdorff Distance |
| ASSD | Average Symmetric Surface Distance |
| BF1 | Boundary F1 Score |

Results are saved as CSV with per-seed values and overall mean +/- std.

## Model Checkpoints

Models are saved at `checkpoints/seed_{seed}/best_model.pth` based on validation Dice score.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sggnet2025,
  title={SGGNet: Integrating Swin Transformer and Dynamic Snake Convolution for FAZ Segmentation in OCTA},
  author={Xue, Guoyi and Deng, Bin},
  journal={PLOS ONE},
  year={2025}
}
```

## License

This code is released under the MIT License.
