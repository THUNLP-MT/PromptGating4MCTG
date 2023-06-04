import sys
from evaluate import load

FILE_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

perplexity = load("perplexity", module_type="metric")
with open(FILE_PATH, 'r') as f:
    lines = f.readlines()
    predictions = [line.strip() for line in lines]
results = perplexity.compute(predictions=predictions, add_start_token=False, model_id='gpt2-large')
with open(OUT_PATH, 'w') as f:
    f.write(str(results))
