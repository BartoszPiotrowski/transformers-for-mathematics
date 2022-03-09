import sys
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam

sys.path.append('scripts')
from data import data_loader
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
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    printf('done')
    printf('loading data...', end=' ')
    train_data = data_loader(tokenizer, args.batch_size, source, target)
    printf('done')
    printf('loading model...', end=' ')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()
    printf('done')
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    printf('training model...')
    train(model, train_data, optimizer, tokenizer, args.train_steps_max,
          save_dir=args.save_dir)
    printf('training done')
