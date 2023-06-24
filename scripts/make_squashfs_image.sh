#!/usr/bin/env zsh
set -ex

TARGET=$1
mksquashfs "$TARGET" "$TARGET.sqfs" -all-root -action 'chmod(o+rX)@!perm(o+rX)'