import sys
from transformers import MBartTokenizer
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

for line in sys.stdin:
	print(' '.join(tokenizer.tokenize(line.strip())))
