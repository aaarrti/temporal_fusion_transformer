#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out
#SBATCH --mem-per-gpu=12G

nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv

cp images/electricity.sqfs /tmp
apptainer run --nv --writable-tmpfs --env-file .env \
  --bind=/tmp/electricity.sqfs:/data:image-src=/ images/image.sif /usr/bin/python scripts/main.py \
  --batch_size=256 --mixed_precision=true --jit_module=true --task=model --experiment=electricity --data_dir=/data
