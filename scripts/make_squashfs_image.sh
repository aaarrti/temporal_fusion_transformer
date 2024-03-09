#!/usr/bin/env zsh

set -ex

TARGET=$1

mkdir -p images/tmp/"$TARGET"

cp -r data/"$TARGET"/training images/tmp/"$TARGET"/
cp -r data/"$TARGET"/validation images/tmp/"$TARGET"/

rm -f images/"$TARGET".sqfs
mksquashfs images/tmp/ images/"$TARGET".sqfs -all-root -action 'chmod(o+rX)@!perm(o+rX)'

rm -r images/tmp/