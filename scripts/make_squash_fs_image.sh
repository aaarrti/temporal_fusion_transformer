#!/usr/bin/env zsh
set -ex
mksquashfs "data" "data.sqfs" -all-root -action 'chmod(o+rX)@!perm(o+rX)'
