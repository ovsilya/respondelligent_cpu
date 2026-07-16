#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#
# NOTE: adjust the paths below to your system before running.
# Run this script from the repository root so that the
# `readvisor.model.*` modules are importable, e.g.:
# bash readvisor/model/scripts/run_mbart_conversion.sh
#

set -e

# --- paths to adjust (placeholders, not real locations) ---
scratch=/home/ovsyannikovilyavl/respondelligent/rg/fastapi_app/app/models/mbart/response_generator
data=/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart

spm_pieces="$data/spm_pieces.txt"
spec_tokens="$data/special_tokens.txt"

# collect list-of-spm-pieces
python -m readvisor.model.collect_spm_pieces \
    $data/train.review $data/train.response \
    $data/valid.review $data/valid.response \
    --spm $scratch/sentencepiece.bpe-2.model \
    --outfile $spm_pieces

echo "saved spm pieces to $spm_pieces"

python -m readvisor.model.collect_special_tokens \
    $data/train.review $data/train.response \
    $data/train.rating $data/train.domain $data/train.est_label \
    $data/valid.rating $data/valid.domain $data/valid.est_label \
    --outfile $spec_tokens

echo "saved special tokens $spec_tokens"

timestamp() {
  date +"%Y-%m-%d" # current time
}

save_pref="mbart_model_$(timestamp)"
outdir=$data/$save_pref/
echo "output path for trimmed mBART model: $outdir"

echo "trimming mBART's embedding matrix..."

python -m readvisor.model.trim_mbart \
    --base_model facebook/mbart-large-cc25 \
    --save_model_to $outdir \
    --reduce-to-vocab $spm_pieces \
    --cache_dir $scratch \
    --add_special_tokens $spec_tokens

echo "done!"
