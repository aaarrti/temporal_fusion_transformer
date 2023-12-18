#!/usr/bin/env zsh

set pipefail

isort $(find . -iregex '.*\(py\)')
pyupgrade --py310-plus $(find . -iregex '.*\(py\)')
black --target-version py310 .