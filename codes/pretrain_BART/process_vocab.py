# convert vocab.json to vocab.txt
import json

VOCAB_JSON = "/path/to/pretrain_BART/vocab.json"
VOCAB_TXT = "/path/to/pretrain_BART/vocab.txt"

with open(VOCAB_JSON, "r", encoding="utf-8") as f:
    vocab = json.load(f)
# sort dict by value
vocab = sorted(vocab.items(), key=lambda item: item[1])

with open(VOCAB_TXT, "w", encoding="utf-8") as f:
    for item in vocab:
        f.write(item[0] + "\n")
