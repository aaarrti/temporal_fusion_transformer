#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out
#SBATCH --mem-per-gpu=12G

apptainer exec --nv images/image.sif python3 temporal_fusion_transformer/main.py \
  --batch_size=512 --mixed_precision=true --jit_module=true --task=model --experiment=electricity
