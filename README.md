# SGGNet: Integrating Swin Transformer and Dynamic Snake Convolution for FAZ Segmentation in OCTA

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
    --lr 1e-4 \
    --batch_size 8

# Train with all 5 seeds
for seed in 42 123 456 789 1024; do
    python train.py --data_root ./data/OCTA-500 --seed $seed
done
```

### Training Details

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine Annealing
- **Loss**: BCE + Dice Loss
- **Input Size**: 512 x 512
- **Batch Size**: 8
- **Epochs**: 100
- **Data Augmentation**: Random horizontal flip, random vertical flip, random 90-degree rotation

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
