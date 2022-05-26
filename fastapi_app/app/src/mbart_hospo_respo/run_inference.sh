#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#
# NOTE: adjust paths below to system before running
#
# Example call:
# bash run_inference.sh 1
#

set -e

GPU=$0
scratch=/srv/scratch6/kew/mbart/hospo_respo/respo_final/

# data=$scratch/data/ # regular test set (2020)
data=/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart
finetuned=/home/ovsyannikovilyavl/respondelligent/rg/fastapi_app/app/models/mbart/response_generator
model_checkpoint=model_28.ckpt
outdir=$data/inference/
outfile=$outdir/translations_28.json

SRC="review_tagged"
TGT="response_tagged"

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $outdir

set -x

python inference.py \
    --model_path $finetuned \
    --checkpoint $model_checkpoint \
    --tokenizer $finetuned \
    --test_source $data/test.$SRC \
    --test_target $data/test.$TGT \
    --infer_target_tags \
    --tags_included \
    --max_output_len 400 \
    --max_input_len 512 \
    --batch_size 4 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 6 \
    --progress_bar_refresh_rate 1 \
    --num_return_sequences 1 \
    --translation $outfile --output_to_json;
