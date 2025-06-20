#!/bin/bash

mode="${1:-release}"
echo "build-mode: $mode" >&2
set -eu

BT=$(git rev-parse --show-toplevel)
cd "$BT"

VENV_DIR="$BT/derived.noindex/venv"
pip_cmd="$VENV_DIR/bin/pip"
py2app_cmd="$VENV_DIR/bin/python3 ./setup-app.py py2app"

# FIXME: not sure which part of setup keeps littering in ./build

rm -rf ./build ./derived.noindex

eval "$(/opt/homebrew/bin/brew shellenv)"
if ! brew bundle check; then
    brew bundle install
fi

./build-model.sh

if ! test -d "$VENV_DIR"; then
    python3.12 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
fi

$pip_cmd uninstall -y mavis-corrector
$pip_cmd install -r requirements.txt

if [ "$mode" == "debug" ]; then
    $pip_cmd install -e .
    py2app_cmd="${py2app_cmd} -v --alias"
else
    $pip_cmd install .
fi

$py2app_cmd
