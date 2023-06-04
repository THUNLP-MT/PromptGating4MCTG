import sys
import nltk

SRC_FILE = sys.argv[1]
TGT_FILE = sys.argv[2]

# parser = nltk.ChunkParserI()

# identify the tense of the sentence
def identify_tense(sentence):
    tense = nltk.pos_tag(nltk.word_tokenize(sentence))
    tense = nltk.ne_chunk(tense)
    return tense

with open(SRC_FILE, 'r') as src, open(TGT_FILE, 'w') as tgt:
    for line in src:
        line = line.strip()
        tense = identify_tense(line)
        print(tense)
