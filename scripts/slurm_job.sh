#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv

cp images/electricity.sqfs /tmp
apptainer run --nv --writable-tmpfs --env-file .env --bind=/tmp/electricity.sqfs:/data:image-src=/ images/image.sif \
  /usr/bin/python scripts/train_model.py --task=model_distributed --experiment=electricity --batch_size=512 \
  --mixed_precision=true --data_dir=/data --verbose=false --profile=false --jit_module=true --epochs=1
