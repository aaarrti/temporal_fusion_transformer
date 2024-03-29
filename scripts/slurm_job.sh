#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv

cp images/electricity.sqfs /tmp
apptainer run --nv --writable-tmpfs --env-file .env --bind=/tmp/electricity.sqfs:/data:image-src=/ images/image.sif \
  /usr/bin/python temporal_fusion_transformer/main.py --task=model --experiment=electricity --batch_size=512 \
  --data_dir=/data --verbose=false --epochs=1
