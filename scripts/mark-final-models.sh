for log in `grep 'training done' exps/*/*.log -l`; do
    data=${log%.*}.data
    if test -d $data; then
        newest_model=$data/`ls -t $data | head -1`
        final_model=${log%.*}.final_model
        if test -d $final_model; then rm -rf $final_model; fi
        echo $newest_model getting copied to $final_model
        cp -r $newest_model $final_model
        if test -d $newest_model/../tokenizer; then
            echo Tokenizer $newest_model/../tokenizer getting copied to $final_model
            cp -r $newest_model/../tokenizer $final_model
        fi
    fi
done
