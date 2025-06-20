#!/bin/bash

# Brew needs to be setup on the target machine.
# xcode-select --install
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# eval "$(/opt/homebrew/bin/brew shellenv)"
# brew analytics off

rhost=macos-sonoma.local
# remote build for corrector plugin.
rexec="ssh $rhost"

set -eu

# For some reason py2app can only target the platform you are running on, so if you need to build
# for an earlier platform, you have to do it inside a VM or something. There might be a better way,
# but haven't found it yet.

src_dir=src/mavis-corrector
dist_dir="derived.noindex/dist/"

$rexec "mkdir -p $src_dir"
rsync -av --delete --exclude *.noindex . $rhost:$src_dir
$rexec "cd $src_dir; ./build.sh"
rsync -a --delete $rhost:$src_dir/$dist_dir/MavisCorrector.plugin ./$dist_dir/sonoma
