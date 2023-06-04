import sys

for ind, line in enumerate(sys.stdin):
    if len(line) > 512:
        print(ind)
