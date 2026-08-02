# FaburNet

FaburNet is a PyTorch model for three-class segmentation of transperineal ultrasound images:

- `0`: background
- `1`: pubic symphysis (PS)
- `2`: fetal head (FH)

The network uses a convolutional/MLP encoder-decoder, token-enhanced decoder blocks, structure-aware uncertainty enhancement, deep supervision, and an exponential moving average (EMA) of the model weights. The evaluation pipeline reports segmentation and angle-of-progression (AoP) measurements.

## Repository structure

```text
FaburNet/
├── train.py                       # Training, validation, EMA, and checkpointing
├── eval.py                        # Checkpoint inference and evaluation CLI
├── models/
│   ├── faburnet.py                # FaburNet decoder enhancements and EMA helper
│   ├── aopnet.py                  # Base encoder-decoder network
│   ├── conv.py                    # Convolution building blocks
│   ├── mytoken.py                 # Token module
│   ├── norm.py                    # Normalization layers
│   └── inference_version/         # Alternative inference-oriented implementation
├── utils/
│   ├── erase_dataloader.py        # Datasets, preprocessing, and augmentation
│   ├── erase.py                   # Point-guided random erasing
│   ├── criterion.py               # Combined Dice/CE training criterion
│   ├── loss.py                    # Loss implementations
│   ├── metric.py                  # Training-time Dice metrics
│   └── meter.py                   # Metric accumulation helpers
├── eval/
│   ├── evaluation.py              # Dice, ASD, HD, and AoP evaluation
│   ├── ellipse.py                 # Ellipse fitting and AoP geometry
│   ├── icc.py                     # ICC/Pearson utilities
│   └── cal_mean.py                # Result aggregation utility
└── dataset/
    └── data/                       # Expected datasets and split files
```

Training creates `checkpoints/` and `tsboardx/`. Evaluation creates `eval_results/` and, when requested, `eval_visualizations/`.

## Installation

Python 3.10 or newer is recommended. Training requires an NVIDIA GPU and a CUDA-enabled PyTorch installation.

Create an environment and install PyTorch using the command appropriate for your CUDA version from the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install torch and torchvision for your CUDA version first, then:
pip install numpy pandas pillow pyyaml tqdm tensorboard tensorboardX \
  SimpleITK opencv-python timm einops batchgenerators
```

Optional utilities require:

```bash
pip install pingouin thop
```

`pingouin` is used by the ICC helper. `thop` is only needed to run the profiling block in `models/faburnet.py` directly.

## Dataset layout

Run all commands from the repository root. The code currently expects the following fixed relative paths:

```text
dataset/data/
├── train/
│   ├── image_mha/
│   │   └── 00001.mha
│   ├── label_mha/
│   │   └── 00001.mha
│   ├── points/
│   │   └── 00001.json
│   └── split/
│       ├── train_fold_0.txt
│       ├── val_fold_0.txt
│       └── ...                    # Folds 1-4 may also be stored here
├── test/
│   ├── test.txt
│   └── testdataset/
│       ├── image_mha/00001.mha
│       └── label_mha/00001.mha
├── psfhs/
│   ├── test.txt
│   ├── img/00001.png
│   └── label/1.png
└── iugc/
    ├── test.txt
    ├── img/00001.png
    └── label/00001.png
```

Each split-file line has the form:

```text
1/MINE/00001
```

The final field is the case ID. Training uses it to find matching `.mha` images, masks, and point annotations. Masks must contain integer class IDs `0`, `1`, and `2`. Training images and masks are expected to be 256×256, and each image must have three channels.

The point JSON files are used by point-guided random erasing. They must contain `fh` and `ps` arrays whose elements are `[x, y]` coordinate pairs. Each case should contain at least 15 FH points and 10 PS points because the augmentation samples up to those counts.

## Training

The default run trains for 400 epochs with AdamW, cosine learning-rate decay, deep supervision, and EMA weights:

```bash
python train.py --name faburnet_fold0
```

A shorter example is:

```bash
python train.py \
  --name faburnet_fold0 \
  --epochs 100 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --scheduler CosineAnnealingLR \
  --min_lr 1e-5 \
  --early_stopping 30
```

Useful options can be listed with:

```bash
python train.py --help
```

During every epoch, the script:

1. trains on `dataset/data/train/split/train_fold_0.txt`;
2. validates on `val_fold_0.txt`;
3. reports performance on `dataset/data/test/test.txt`;
4. saves a checkpoint when mean validation Dice over PS and FH improves.

Outputs are written to:

```text
checkpoints/<run-name>/config.yml   # Saved CLI configuration
checkpoints/<run-name>/log.csv      # Epoch metrics
checkpoints/<run-name>/*.pth        # Best checkpoint(s)
tsboardx/<run-name>/                # TensorBoard event files
```

View the training curves with:

```bash
tensorboard --logdir tsboardx
```

### Current training constraints

- `train.py` selects physical GPU `0` through `CUDA_VISIBLE_DEVICES` and calls `.cuda()` directly, so CPU training is not supported without changing the script.
- Batch size `8`, fold `0`, image size `256`, and data-loader worker count `8` are currently fixed inside `get_dataloader()`.
- The model is fixed to three input channels and three output classes. The parsed `--input_w`, `--input_h`, `--input_channels`, `--num_classes`, and `--arch` values do not currently reconfigure the model or loaders.
- Only `AdamW` is implemented even though the CLI help mentions other optimizers.
- Keep deep supervision enabled for the current `FaburNet` output format.

To use another fold, batch size, image size, or GPU, update `get_dataloader()` and the `CUDA_VISIBLE_DEVICES` assignment near the top of `train.py`.

## Evaluation

Evaluate a saved checkpoint on all three supported datasets:

```bash
python eval.py \
  --checkpoint checkpoints/faburnet_fold0/your_checkpoint.pth
```

Evaluate selected datasets only:

```bash
python eval.py \
  --checkpoint checkpoints/faburnet_fold0/your_checkpoint.pth \
  --datasets test psfhs \
  --batch-size 1 \
  --num-workers 4
```

Save prediction overlays with fitted ellipses and AoP geometry:

```bash
python eval.py \
  --checkpoint checkpoints/faburnet_fold0/your_checkpoint.pth \
  --datasets test psfhs iugc \
  --save-visualizations
```

By default, evaluation loads the `ema_model` weights stored by training. Use `--weights model` to evaluate the non-EMA weights:

```bash
python eval.py \
  --checkpoint checkpoints/faburnet_fold0/your_checkpoint.pth \
  --weights model
```

## Quick start

```bash
# 1. Activate the environment and verify that the dataset follows the layout above.
source .venv/bin/activate

# 2. Train.
python train.py --name faburnet_fold0 --epochs 400

# 3. Select the best checkpoint printed during training and evaluate it.
python eval.py \
  --checkpoint checkpoints/faburnet_fold0/your_checkpoint.pth \
  --datasets test psfhs iugc \
  --save-visualizations
```
