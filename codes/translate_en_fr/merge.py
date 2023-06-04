import sys

TGT_FILE = sys.argv[1]
TOT_LINES = int(sys.argv[2])

with open(TGT_FILE, "w") as tgt_file:
    for i in range(0, 4500000, TOT_LINES):
        with open(TGT_FILE + ".{}".format(i), "r") as src_file:
            for line in src_file:
                tgt_file.write(line)
