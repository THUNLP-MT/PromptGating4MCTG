import sys

with open('/home/lzj/lzj/plug4MSG/data/wmt16ende/train.de.shortind', 'r') as f:
    sample = [int(x) for x in f.readlines()]

for ind, line in enumerate(sys.stdin):
    if ind not in sample:
        print(line, end='')
