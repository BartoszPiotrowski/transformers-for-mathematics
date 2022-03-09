#!/bin/bash

gpu=$1
dir=`dirname $0`
commit=`git rev-parse --short HEAD`
log=${0%.*}-$commit-$RANDOM.log
data=${log%.*}.data

export CUDA_VISIBLE_DEVICES=$gpu
source venv-python3.8/bin/activate

python3 $dir/untrained.py \
    --train_data data/simple/multiplication/train_len_5 \
    --lr 1e-5 \
    --save_dir $data \
    | tee $log

