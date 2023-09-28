#!/usr/bin/env zsh

set -ex

# shellcheck disable=SC2046
isort $(find . -iregex '.*\(py\)')