import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

for line in sys.stdin:
    line = line.strip().split()
    # decode
    print(tokenizer.decode(tokenizer.convert_tokens_to_ids(line)))
