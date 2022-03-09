for m in `find exps -name '*final_model'`; do
    if test ! -s $m.pred.acc; then
        ./scripts/eval.sh $m
    fi
done
