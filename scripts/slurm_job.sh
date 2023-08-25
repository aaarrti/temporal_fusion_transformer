#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --mem-per-gpu=12G

nvidia-smi
cp images/electricity.sqfs /tmp
apptainer run --nv --writable-tmpfs --env-file .env \
  --bind=/tmp/electricity.sqfs:/data:image-src=/ images/image.sif /usr/bin/python scripts/main.py \
  --batch_size=1024 --mixed_precision=true --jit_module=true --task=model --experiment=electricity --data_dir=/data
