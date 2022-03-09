import os
import torch
from random import shuffle
from utils import read_lines
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

UNK = '<unk>'
PAD = '<pad>'
EOS = '<eos>'
BOS = '<pad>'

class Data(Dataset):

    def __init__(self, tokenizer, source, target=None):

        self.tokenizer = tokenizer
        self.sources = read_lines(source)
        self.source_len = max([len(l.split(' ')) for l in self.sources])
        if target:
            self.targets = read_lines(target)
            self.target_len = max([len(l.split(' ')) for l in self.targets])
            assert len(self.sources) == len(self.targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        '''return the input ids, attention masks and target ids'''

        source = self.tokenizer(
            [self.sources[i]],
            max_length=self.source_len + 10,
            pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        if self.targets:
            target = self.tokenizer(
                [self.targets[i]],
                max_length=self.target_len + 10,
                pad_to_max_length=True,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            target_ids = target['input_ids'].squeeze()
            target_mask = target['attention_mask'].squeeze()

            if i + 1 == len(self):
                # if we arrived at the end of examples, permute
                sources_targets = list(zip(self.sources, self.targets))
                shuffle(sources_targets)
                self.sources, self.targets = zip(*sources_targets)

            return {
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
                'target_ids': target_ids.to(dtype=torch.long),
                'target_mask': target_mask.to(dtype=torch.long),
            }

        else:
            return {
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
            }

class DataLM(Dataset):

    def __init__(self, tokenizer, source, target=None):
        self.sep = '='
        self.tokenizer = tokenizer
        self.sources = read_lines(source)
        self.source_len = max([len(l.split(' ')) for l in self.sources])
        if target:
            self.targets = read_lines(target)
            self.target_len = max([len(l.split(' ')) for l in self.targets])
            assert len(self.sources) == len(self.targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        '''return the input ids, attention masks and target ids'''

        if self.targets:
            example = self.tokenizer(
                [self.sources[i] + ' ' + self.sep + ' ' + self.targets[i]],
                max_length=self.target_len + self.source_len + 10,
                pad_to_max_length=True,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            if i + 1 == len(self):
                # if we arrived at the end of examples, permute
                sources_targets = list(zip(self.sources, self.targets))
                shuffle(sources_targets)
                self.sources, self.targets = zip(*sources_targets)

        else:
            example = self.tokenizer(
                [self.sources[i] + ' ' + self.sep],
                max_length=self.target_len + self.source_len + 10,
                pad_to_max_length=True,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

        example_ids = example['input_ids'].squeeze()
        example_mask = example['attention_mask'].squeeze()

        return {
            'example_ids': example_ids.to(dtype=torch.long),
            'example_mask': example_mask.to(dtype=torch.long),
        }


def data_loader(tokenizer, batch_size, source, target=None):
    return DataLoader(Data(tokenizer, source, target), batch_size)

def whitespace_tokenizer(data_path, save_dir=None):
    source = read_lines(data_path + '.src')
    target = read_lines(data_path + '.tgt')
    vocab = {PAD, BOS, EOS, UNK}
    for l in source + target:
        vocab.update(l.split(' '))
    vocab = sorted(list(vocab))
    vocab = {vocab[i]: i for i in range(len(vocab))}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token=UNK))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single=f'$0 {EOS}',
        special_tokens=[(EOS, tokenizer.token_to_id(EOS))])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        #pad_token=PAD, bos_token=BOS, eos_token=EOS,
    )
    tokenizer.add_special_tokens(
        {'pad_token': PAD, 'eos_token': EOS, 'bos_token': BOS}
    )
    save_path = os.path.join(save_dir, 'tokenizer')
    tokenizer.save_pretrained(save_path)
    print(f'tokenizer prepared and saved at {save_path}')
    return tokenizer

def load_tokenizer(path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    return tokenizer
