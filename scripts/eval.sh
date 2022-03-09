final_model=$1
gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu
source venv-python3.8/bin/activate
sh=`echo $final_model | cut -d'-' -f1-2`.sh
train_data=`grep train_data $sh | rev | cut -d' ' -f2 | rev`
test_data=`echo $train_data | sed 's/train/test/g'`.src
echo Evaluating $final_model on $test_data
if echo $final_model | grep pretrained; then
    python3 $final_model/../pretrained-eval.py \
        --model $final_model \
        --test_data $test_data \
        > $final_model.pred
fi
if echo $final_model | grep untrained; then
    python3 $final_model/../untrained-eval.py \
        --model $final_model \
        --tokenizer $final_model/tokenizer \
        --test_data $test_data \
        > $final_model.pred
fi
echo Predictions saved to $final_model.pred
python3 scripts/accuracy.py \
    ${test_data%.*}.tgt \
    $final_model.pred \
    > $final_model.pred.acc
