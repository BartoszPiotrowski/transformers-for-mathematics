import sys
import argparse
from transformers import T5ForConditionalGeneration, T5Config
from torch.optim import Adam

sys.path.append('scripts')
from data import data_loader, whitespace_tokenizer
from model import train
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
        default=1e-5,
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


    printf('preparing tokenizer...')
    tokenizer = whitespace_tokenizer(args.train_data, args.save_dir)
    printf('loading data...', end=' ')
    source = args.train_data + '.src'
    target = args.train_data + '.tgt'
    train_data = data_loader(tokenizer, args.batch_size, source, target)
    printf('done')
    printf('initializing model...', end=' ')
    config = T5Config(
        decoder_start_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = T5ForConditionalGeneration(config).cuda()
    printf('done')
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    printf('training model...')
    train(model, train_data, optimizer, tokenizer, args.train_steps_max,
          save_dir=args.save_dir)
    printf('training done')
