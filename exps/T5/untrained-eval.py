import sys
import argparse
from transformers import T5ForConditionalGeneration

sys.path.append('scripts')
from data import data_loader, load_tokenizer
from model import eval
from utils import eprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str)
    parser.add_argument(
        '--tokenizer',
        type=str)
    parser.add_argument(
        '--test_data',
        type=str)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    args = parser.parse_args()

    eprint('loading tokenizer...', end=' ')
    tokenizer = load_tokenizer(args.tokenizer)
    eprint('done')
    eprint('loading data...', end=' ')
    test_data = data_loader(tokenizer, args.batch_size, args.test_data)
    eprint('done')
    eprint('loading model...', end=' ')
    model = T5ForConditionalGeneration.from_pretrained(f'{args.model}').cuda()
    eprint('done')
    eprint('decoding...')
    eval(model, test_data, tokenizer)
    eprint('done')
