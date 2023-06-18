#!/usr/bin/env bash
#$ -binding linear:2  # request 2 cpus
#$ -l cuda=1          # request one GPU
#$ -l h_vmem=14G
#$ -l mem_free=14G
#$ -cwd               # change working directory (to current)
#$ -N ex              # set consistent base name for output and error files
#$ -V                 # provide environment variables
#$ -e logs/err/
#$ -o logs/out/
set -ex
IMAGE_DIR="/home/artem/shared/nv_tf_py38"
##########################################################################
# Reinstall model source code
##########################################################################
apptainer exec --contain --bind "${IMAGE_DIR}/venv.img:/venv:image-src=/" \
  "${IMAGE_DIR}/image.sif" /venv/bin/python -m pip install 'git+https://github.com/aaarrti/tf2_temporal_fusion_transformer.git@dev' --force-reinstall --no-deps
##########################################################################
# Actual script
##########################################################################
apptainer exec --nv --env-file env --contain \
  --bind datasets.sqfs:/datasets:image-src=/ \
  --bind "${IMAGE_DIR}/venv.img:/venv:image-src=/,ro" \
  --bind scripts/:/scripts \
  --bind logs/:/logs \
  "${IMAGE_DIR}/image.sif" /venv/bin/python /scripts/train_keras_model.py \
  --experiment=electricity --data_dir=/datasets --logs_dir=/logs --batch_size=512 --epochs=5
##########################################################################
