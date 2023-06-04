import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

for line in sys.stdin:
	print(' '.join(tokenizer.tokenize(line.strip())))
