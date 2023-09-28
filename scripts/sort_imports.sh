#!/usr/bin/env zsh

set -ex

isort $(find . -iregex '.*\(py\)')