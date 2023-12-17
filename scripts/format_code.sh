#!/usr/bin/env zsh

set -ex pipefail

isort "$(find temporal_fusion_transformer -iregex '.*\(py\)')"
pyupgrade --py310-plus "$(find temporal_fusion_transformer -iregex '.*\(py\)')"
black --target-version py310 temporal_fusion_transformer