import sys

LENGTH_INTERVAL = 10
LENGTHS = [(i, i + LENGTH_INTERVAL) for i in range(20, 200, LENGTH_INTERVAL)]
LENGTHS = [(0, 20)] + LENGTHS + [(200, 10000)]

all_len = [[l]*20 for l in LENGTHS]
all_len = [item for sublist in all_len for item in sublist]

FILE = sys.argv[1]
RESULT = sys.argv[2]

acc = 0
with open(FILE) as f:
    with open(RESULT, "w") as fr:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for lid, line in enumerate(lines):
            length = len(line.split())
            start, end = all_len[lid]
            if start <= length < end:
                acc += 1
        fr.write("Accuracy: {}".format(acc / len(lines)))
