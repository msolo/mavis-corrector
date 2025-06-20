#!/bin/bash

set -eu

BT=$(git rev-parse --show-toplevel)
cd "$BT"

VENV_DIR="$BT/derived.noindex/venv-model"
pip_cmd="$VENV_DIR/bin/pip"

model_path="$BT/models/t5-small-spoken-typo.ct2"

if test -d "$model_path"; then
    echo "Model $model_path exists, skipping." >&2
    exit 0
fi

if ! test -d "$VENV_DIR"; then
    python3.12 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
fi

$pip_cmd install -r requirements-model.txt

source "$VENV_DIR/bin/activate"

mkdir -p models
ct2-transformers-converter --model willwade/t5-small-spoken-typo --output_dir $model_path
./download-tokenizer.py --model willwade/t5-small-spoken-typo --output_dir $model_path
