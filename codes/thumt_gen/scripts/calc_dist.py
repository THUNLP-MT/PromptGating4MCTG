import sys

FILE_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]
PREFIX_PATH = "/home/lzj/lzj/plug4MSG/data/yelp/infer/pre_tokens.25.txt"

def count_ngram(text_samples, n, tokenizer=None):
    """
    Count the number of unique n-grams
    :param text_samples: list, a list of samples
    :param n: int, n-gram
    :return: the number of unique n-grams in text_samples
    """
    if len(text_samples) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    ngram = set()
    for sample in text_samples:
        if len(sample) < n:
            continue

        sample = list(map(str, sample))
        for i in range(len(sample) - n + 1):
            ng = ' '.join(sample[i: i + n])
            # print(ng)

            ngram.add(' '.join(ng))
    return len(ngram)

with open(PREFIX_PATH, 'r') as f:
    prefixes = f.readlines()
    prefixes = [line.strip() for line in prefixes]

# calculate average of dist-1, dist-2, dist-3
with open(FILE_PATH, 'r') as f:
    lines = f.readlines()
    predictions = [line.strip() for line in lines]
    
for i in range(len(predictions)):
    if predictions[i][0:len(prefixes[i])] != prefixes[i]:
        raise ValueError("prefixes don't match")
    predictions[i] = predictions[i][len(prefixes[i]):]
predictions = [line.split() for line in predictions]
    
total_tokens = sum([len(line) for line in predictions])
dist1 = count_ngram(predictions, 1) / total_tokens
dist2 = count_ngram(predictions, 2) / total_tokens
dist3 = count_ngram(predictions, 3) / total_tokens
dist = (dist1 + dist2 + dist3) / 3

with open(OUT_PATH, 'w') as f:
    f.write(str(dist1) + ' ' + str(dist2) + ' ' + str(dist3) + '\n')
    f.write(str(dist))
