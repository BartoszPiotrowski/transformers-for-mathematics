import sys
import argparse
from transformers import GPT2LMHeadModel

sys.path.append('scripts')
from data_lm import eval_data_loader, load_tokenizer
from model_lm import eval
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
    args = parser.parse_args()

    eprint('loading tokenizer...', end=' ')
    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    eprint('done')
    eprint('loading data...', end=' ')
    test_data = eval_data_loader(tokenizer, args.test_data)
    eprint('done')
    eprint('loading model...', end=' ')
    model = GPT2LMHeadModel.from_pretrained(f'{args.model}').cuda()
    model.config.pad_token_id = model.config.eos_token_id
    eprint('done')
    eprint('decoding...')
    eval(model, test_data, tokenizer)
    eprint('done')
