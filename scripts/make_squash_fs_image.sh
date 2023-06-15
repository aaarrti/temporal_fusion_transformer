#!/usr/bin/env zsh
set -ex
mksquashfs "data/$1" "data/$1.sqfs" -all-root -action 'chmod(o+rX)@!perm(o+rX)'
