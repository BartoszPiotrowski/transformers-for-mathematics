#!/bin/bash

gpu=$1
dir=`dirname $0`
commit=`git rev-parse --short HEAD`
log=${0%.*}-$commit-$RANDOM.log
data=${log%.*}.data

export CUDA_VISIBLE_DEVICES=$gpu
source venv-python3.8/bin/activate

python3 $dir/pretrained.py \
    --train_data data/mathematics/numbers__base_conversion_train.char_num \
    --lr 1e-5 \
    --batch_size 32 \
    --train_steps_max 200000 \
    --save_dir $data \
    | tee $log

