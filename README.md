# HAQ-MobileNetV3-Classify

This repository provides a PyTorch implementation of HAQ-style automated
quantization and finetuning, with examples for MobileNetV2 and MobileNetV3
and a DTD dataset setup. It includes:

![HAQ Overview](docs/HAQ-Overview.png)

- Pretraining and finetuning scripts.
- DDPG-based RL search for mixed-precision policies.
- K-means weight quantization and linear mixed-precision quantization.
- A lookup-table-based latency cost model for linear quantization.

## Project Layout

- `scripts/pretrain.py`: Train a full-precision model.
- `scripts/rl_quantize.py`: RL search for quantization strategy.
- `scripts/finetune.py`: Finetune a model with a fixed quantization strategy.
- `lib/env/quantize_env.py`: K-means quantization environment.
- `lib/env/linear_quantize_env.py`: Linear mixed-precision environment.
- `lib/rl/*`: DDPG implementation and replay buffer.
- `lib/utils/*`: Data loading, quantization utilities, and profiling helpers.
- `models/*`: MobileNet/MobileNetV2/MobileNetV3 definitions.
- `run/*.sh`: Example scripts for common workflows.
- `run/dtd/*.sh`: DTD-specific example scripts (MobileNetV3).
- `docs/`: Reference papers and notes.

## Requirements

See `requirements.txt`.

## Dataset Setup (DTD example)

This repo assumes a DTD-style folder for raw images:

```
data/dtd/images/<class_name>/*.jpg
```

Run the split script:

```
python lib/utils/make_data.py
```

It will create:

```
data/DTD/train/<class_name>/*.jpg
data/DTD/val/<class_name>/*.jpg
```

## Pretrained Weights (optional)

Download MobileNetV3 weights:

```
bash run/setup.sh
```

This script installs requirements, prepares folders, and downloads
`mobilenetv3small-f3be529c.pth`.

## Workflow Overview

There are two main quantization workflows:

1) K-means weight quantization (model size objective).
2) Linear mixed-precision quantization (latency objective).

Both use `scripts/rl_quantize.py` to search a per-layer policy and `scripts/finetune.py`
to train with the selected policy.

## Workflow A: K-means Quantization (model size)

### 1. Search

```
bash run/run_kmeans_quantize_search.sh
```

This runs `scripts/rl_quantize.py` with `QuantizeEnv` (k-means). The RL agent outputs
a per-layer bit strategy for weights.

### 2. Apply Strategy + Finetune

Open `scripts/finetune.py` and update the `strategy` list for your model in the
`if args.arch.startswith('resnet50'):` block (or add your own block).

Then run:

```
bash run/run_kmeans_quantize_finetune.sh
```

### 3. Evaluate

```
bash run/run_kmeans_quantize_eval.sh
```

## Workflow B: Linear Mixed-Precision (latency)

This uses `LinearQuantizeEnv` and a lookup table for latency costs.

### 1. Search

```
bash run/run_linear_quantize_search.sh
```

The lookup table is loaded from:

```
lib/simulator/lookup_tables/
```

### 2. Apply Strategy + Finetune

Update the `strategy` list for your model in the `if args.linear_quantization:`
block inside `scripts/finetune.py` (currently only `qmobilenetv2` is provided).

Then run:

```
bash run/run_linear_quantize_finetune.sh
```

### 3. Evaluate

```
bash run/run_linear_quantize_eval.sh
```

## Key Arguments

Common options in `scripts/rl_quantize.py`:

- `--arch`: model name (see `models/__init__.py`).
- `--dataset` and `--dataset_root`: dataset selection and path.
- `--preserve_ratio`: compression target.
- `--min_bit`, `--max_bit`: bitwidth range.
- `--linear_quantization`: enable linear mixed-precision search.
- `--train_size`, `--val_size`: subsample sizes for faster search.

Common options in `scripts/finetune.py`:

- `--linear_quantization`: use QConv2d/QLinear mixed precision.
- `--pretrained`: load pretrained weights (see `models/*`).

## Notes and Caveats

- `run/setup.sh` uses `wget` and requires network access.
- DTD examples are under `run/dtd/`.
- `linear_quantize_env.py` uses a lookup table for latency. Add your own
  table if you change models or hardware assumptions.
- `scripts/finetune.py` expects you to manually paste the searched strategy.

## Example Results (from original project notes)

| Models                   | preserve ratio | Top1 Acc (%) | Top5 Acc (%) |
| ------------------------ | -------------- | ------------ | ------------ |
| resnet50 (original)      | 1.0            | 76.15        | 92.87        |
| resnet50 (10x compress)  | 0.1            | 75.48        | 92.42        |

| Models                     | preserve ratio | Top1 Acc (%) | Top5 Acc (%) |
| -------------------------- | -------------- | ------------ | ------------ |
| mobilenetv2 (original)     | 1.0            | 72.05        | 90.49        |
| mobilenetv2 (0.6x latency) | 0.6            | 71.23        | 90.00        |

## License

MIT, see `LICENSE`.
