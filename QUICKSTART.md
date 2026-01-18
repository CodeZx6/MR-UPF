# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Place your data in the following structure:
```
data/
├── TaxiBJ/
│   ├── train/
│   │   ├── basis.npy
│   │   ├── time_correlation.npy
│   │   ├── time_c_feature.npy
│   │   └── time_text.npy
│   └── test/
│       └── (same structure as train)
├── 2_data.npy
├── 5_data.npy
└── n_data.npy
```

## Training

Basic training:
```bash
python train.py --dataset TaxiBJ --batch_size 128
```

Advanced configuration:
```bash
python train.py \
    --dataset TaxiBJ \
    --batch_size 128 \
    --learning_rate 2e-4 \
    --base_channels 128 \
    --num_epochs 200 \
    --beta_coefficient 0.5 \
    --val_interval 20 \
    --lr_decay_epoch 50
```

## Key Arguments

- `--base_channels`: Controls model capacity (default: 128)
- `--beta_coefficient`: Graph convolution sparsity threshold (default: 0.5)
- `--learning_rate`: Initial learning rate (default: 2e-4)
- `--val_interval`: Epochs between validation runs (default: 20)
- `--lr_decay_epoch`: Learning rate halving interval (default: 50)

## Output

Results are saved to:
- Checkpoints: `./checkpoints/<dataset>_<channels>_<timestamp>/`
- TensorBoard logs: `./runs/<dataset>_<channels>_<timestamp>/`
- Results: `results.txt` in checkpoint directory

## Monitoring

```bash
tensorboard --logdir=./runs
```

## Model Architecture

The framework uses a hierarchical approach:
1. **Semantic Encoding**: Multi-scale textual features
2. **Graph Convolution**: Adaptive spatial dependencies
3. **Temporal Modeling**: GRU-based sequential processing
4. **Cross-Modal Fusion**: Attention-based integration

## Troubleshooting

**CUDA out of memory**: Reduce `--batch_size`

**Slow training**: Increase `--val_interval`

**Poor convergence**: Adjust `--learning_rate` or `--beta_coefficient`
