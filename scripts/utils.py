import sys


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read().splitlines()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def printf(*args, **kwargs):
    print(*args, flush=True, **kwargs)
