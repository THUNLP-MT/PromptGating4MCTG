#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections


def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")

    parser.add_argument("--corpus", help="input corpus")
    parser.add_argument("--input", default="vocab.txt",
                        help="Input vocabulary path")

    return parser.parse_args()

def get_vocab(filename):
    vocab = set()
    
    with open(filename, "rb") as fd:
        for line in fd:
            vocab.add(line.strip())
    
    return vocab
    

def count_words(filename):
    counter = set()

    with open(filename, "rb") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    return counter


def control_symbols(string):
    if not string:
        return []
    else:
        symbs = string.strip().split(",")
        return [sym.encode("ascii") for sym in symbs]


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, _ = list(zip(*pairs))

    with open(name, "wb") as f:
        for word in words:
            f.write(word)
            f.write("\n".encode("ascii"))


def main(args):
    vocab = get_vocab(args.input)

    words = count_words(args.corpus)

    # calculate coverage
    coverage = len(words & vocab) / len(words)

    print("Total vocab: %d" % len(vocab))
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f%%" % (coverage * 100))


if __name__ == "__main__":
    main(parse_args())
