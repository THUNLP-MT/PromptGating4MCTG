import sys

seps = [', ', '. ', '? ', '! ', '; ', ': ', ',)', '.)', '?)', '!)', ';)', '."', ".'", '?"', "?'", '!"', "!'", ',"', ",'", ':"', ":'", ';"', ";'", '...', '- ']

for ind, line in enumerate(sys.stdin):
    line = line.strip()
    if line[-1] not in ['.', '?', '!', ';', '"', "'"]:
        line += '.'
    line += ' '
    
    # calculate number of clauses
    num_clause = 0
    for sep in seps:
        num_clause += line.count(sep)
    
    print("The number of clauses is ", num_clause, ".")
