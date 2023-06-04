from math import ceil
import sys
from random import sample, randint
from gensim.parsing.preprocessing import remove_stopwords

# MASK_RATE = 0.3
MAX_MASK_NUM = 3
MIN_MASK_NUM = 1

# cnt = 0

for ind, line in enumerate(sys.stdin):
    words = list(map(lambda x: x.strip(), line.split(' ')))
    
    clr_words = remove_stopwords(' '.join(words))
    clr_words = list(map(lambda x: x.strip(), clr_words.split(' ')))
    
    if len(clr_words) == 0:
    
        if len(words) < MIN_MASK_NUM:
            sample_num = len(words)
        elif len(words) <= MAX_MASK_NUM:
            sample_num = randint(MIN_MASK_NUM, len(words))
        else:
            sample_num = randint(MIN_MASK_NUM, MAX_MASK_NUM)

        res = sorted(sample(range(len(words)), sample_num))
        out = [words[i] for i in res]
    
    else:
        
        if len(clr_words) < MIN_MASK_NUM:
            sample_num = len(clr_words)
        elif len(clr_words) <= MAX_MASK_NUM:
            sample_num = randint(MIN_MASK_NUM, len(clr_words))
        else:
            sample_num = randint(MIN_MASK_NUM, MAX_MASK_NUM)

        res = sorted(sample(range(len(clr_words)), sample_num))
        out = [clr_words[i] for i in res]
    print(*out)
    
    # cnt += 1
    # if cnt > 500:
    #     break
