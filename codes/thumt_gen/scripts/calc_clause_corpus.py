import sys

CLAUSES = sys.argv[1]
RESULT = sys.argv[2]

seps = [', ', '. ', '? ', '! ', '; ', ': ', ',)', '.)', '?)', '!)', ';)', '."', ".'", '?"', "?'", '!"', "!'", ',"', ",'", ':"', ":'", ';"', ";'", '...', '- ']

acc, tot = 0, 0
with open(CLAUSES) as fc:
    with open(RESULT) as fr:
        fcall = fc.readlines()
        for clause in fcall:
            num_clause_given = int(list(map(lambda x: x.strip(), clause.split(' ')))[-2])
            rline =  fr.readline().strip()
            if rline[-1] not in ['.', '?', '!', ';', '"', "'"]:
                rline += '.'
            rline += ' '
            # calculate number of clauses
            num_clause = 0
            for sep in seps:
                num_clause += rline.count(sep)

            tot += 1
            if num_clause == num_clause_given:
                acc += 1

print("Clause Accuracy:", acc / tot)
