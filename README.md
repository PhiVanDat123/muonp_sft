# SFT, DPO, SPFT, and TDPO Training

This repository contains four training recipes:

- **SFT**: supervised fine-tuning.
- **DPO**: direct preference optimization.
- **SPFT**: stable preference fine-tuning using the Eq. 35 token-weighted SFT loss.
- **TDPO**: token-level direct preference optimization with sequential forward-KL.

All recipes use **AdamW** via `optim: adamw_torch`.

Datasets:

- **SFT**: `hkust-nlp/deita-10k-v0`
- **DPO/SPFT/TDPO**: `HuggingFaceH4/ultrafeedback_binarized`

## Setup

Install the dependencies in your Python environment:

```bash
pip install -r requirments.txt
```

Adjust model paths, dataset names, batch sizes, and output directories in:

```text
recipes/llama3.2-1b-deita-dpomix/sft.yaml
recipes/llama3.2-1b-deita-dpomix/dpo.yaml
recipes/llama3.2-1b-deita-dpomix/spft.yaml
recipes/llama3.2-1b-deita-dpomix/tdpo.yaml
```

## Run SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/sft.yaml
```

## Run DPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/dpo.yaml
```

## Run SPFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/spft.yaml
```

## Run TDPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/tdpo.yaml
```
