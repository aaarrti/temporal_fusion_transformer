#!/usr/bin/env bash
#$ -binding linear:2  # request 2 cpus
#$ -l cuda=1          # request one GPU
#$ -l h_vmem=8G
#$ -l mem_free=8G
#$ -cwd               # change working directory (to current)
#$ -N ex              # set consistent base name for output and error files
#$ -V                 # provide environment variables
#$ -e err_logs/
#$ -o logs/
set -ex
IMAGE_DIR="/home/artem/shared/nv_tf_py38"
# Reinstall model source code
# apptainer exec -B "${IMAGE_DIR}/venv.img:/venv:image-src=/" "${IMAGE_DIR}/IMAGE.sif" /venv/bin/python -m pip install . --force-reinstall --no-deps
############################################################################
# Actual script
##########################################################################
apptainer exec --nv --env-file .env --contain --bind data.sqfs:/data:image-src=/ --bind /home/artem/temporal_fusion_transformer/scripts/:/ \
  "${IMAGE_DIR}/image.sif" python /scripts/train_keras_model.py --experiment=electricity --data_path=/data --batch_size=512
