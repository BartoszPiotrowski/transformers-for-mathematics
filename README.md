# Transformers for mathematics

Data and scripts for training (or fine-tuning) and evaluating transformer models
to perform various mathematical tasks.


## Requirements

* CUDA-capable GPU
* Python packages listed in `venv-python3.8.txt` . To install them run
  ```
  pip install -r venv-python3.8.txt
  ```

## Training

To train a transformer model on a given mathematical task, you need to run
an appropriate bash script from `exps`. For instance, to train GPT2 from scratch
on addition data, run:

```
./exps/GPT2/untrained-addition.sh 0
```

The parameter `0` at the end indicates the we want to use GPU number `0`.

## Evaluating

The training script creates a directory
`exps/GPT2/untrained-addition-*.final_model`
where a trained model is saved. To evaluate it, run:

```
./scripts/eval.sh exps/GPT2/untrained-addition-*.final_model 0
```
