#!/usr/bin/env zsh

set -ex
mkdir -p thrirdparty
cd thrirdparty
git clone --depth 1 --filter=blob:none https://github.com/google-research/google-research.git --sparse
cd google-research
git sparse-checkout init --cone
git sparse-checkout set tft
rm -r -f .git
tf_upgrade_v2 --intree tft/ --outtree tft/ --reportfile report.txt
# Need to add tf.compat.v1.experimental.output_all_intermediates(True)

