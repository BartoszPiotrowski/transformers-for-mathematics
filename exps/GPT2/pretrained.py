import sys
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import Adam

sys.path.append('scripts')
from data_lm import data_loader, eval_data_loader
from model_lm import train
from utils import printf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data',
        type=str)
    parser.add_argument(
        '--save_dir',
        type=str)
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float)
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int)
    parser.add_argument(
        '--train_steps_max',
        default=100000,
        type=int)
    args = parser.parse_args()

    source = args.train_data + '.src'
    target = args.train_data + '.tgt'
    printf('loading tokenizer...', end=' ')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    printf('done')
    printf('loading data...', end=' ')
    train_data = data_loader(tokenizer, args.batch_size, source, target)
    valid_data = eval_data_loader(tokenizer, source, target) # batch size = 1
    printf('done')
    printf('loading model...', end=' ')
    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    model.config.pad_token_id = model.config.eos_token_id
    printf('done')
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    printf('training model...')
    train(model, train_data, valid_data, optimizer, tokenizer, args.train_steps_max,
          save_dir=args.save_dir)
    printf('training done')
