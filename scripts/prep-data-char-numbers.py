import sys
from random import shuffle

def append_line(line, filename):
    with open(filename, encoding='utf-8', mode='a') as f:
        f.write(line + '\n')

def tokenize(line):
    conns_orig = ['**']
    conns = [' '.join(conn) for conn in conns_orig]
    conns_dict = dict(zip(conns, conns_orig))
    line_list = []
    line_token = ''
    for ch in line:
        if ch == ' ':
            if line_token:
                line_list.append(line_token)
                line_token = ''
        elif str.isalpha(ch):
            line_token += ch
        else:
            if line_token:
                line_list.append(line_token)
                line_token = ''
            line_list.append(ch)
    if line_token:
        line_list.append(line_token)
    out_str = ' '.join(line_list)
    for conn in conns:
        if conn in out_str:
            out_str = out_str.replace(conn, conns_dict[conn])
    return out_str

file = sys.argv[1]
file_src = file.replace('.txt', '.char_num.src')
open(file_src, 'w').close()
file_tgt = file.replace('.txt', '.char_num.tgt')
open(file_tgt, 'w').close()

with open(file) as f:
    lines = f.read().splitlines()

lines = [tokenize(l) for l in lines]

assert len(lines) % 2 == 0
permut = list(range(int(len(lines)/2)))
shuffle(permut)
for i in permut:
    append_line(lines[2*i], file_src)
    append_line(lines[2*i+1], file_tgt)
