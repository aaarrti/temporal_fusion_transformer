#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv images/image.sif \
  python3 temporal_fusion_transformer/main.py --batch_size=128 --mixed_precision=true --jit_module=true
