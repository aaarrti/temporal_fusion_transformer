#!/usr/bin/env zsh

set -ex
mkdir -p thrirdparty
cd thrirdparty
git clone --depth 1 --filter=blob:none https://github.com/google-research/google-research.git --sparse
cd google-research
git sparse-checkout init --cone
git sparse-checkout set tft
rm -r -f .git

git clone # TODO transformer engine

git clone --depth 1 --filter=blob:none  https://github.com/NVIDIA/DALI.git --sparse
cd DALI
git sparse-checkout init --cone
git sparse-checkout set dali_tf_plugin
git sparse-checkout init --cone

