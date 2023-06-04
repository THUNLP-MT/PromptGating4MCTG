import sys

orig_vocab=open(sys.argv[1], 'r').readlines()

with open(sys.argv[2], 'r') as fd:
    small_vocab=fd.readlines()
    small_vocab=[v.strip() for v in small_vocab]

_set = set()
for line in sys.stdin:
	line=line.strip().split()
	for word in line:
		_set.add(word)

reverse_dict = {}

for i in range(len(orig_vocab)):
    reverse_dict[orig_vocab[i].strip()] = i

map_list = open(sys.argv[3], 'w')
new_small_list = open(sys.argv[4], 'w')

for word in small_vocab:
    if word in reverse_dict:
        map_list.write("%d\n" % reverse_dict[word])
        new_small_list.write("%s\n" % word)

# with open(sys.argv[2], 'a') as fd:
for word in _set:
    if word not in small_vocab and word in reverse_dict:
        new_small_list.write("%s\n" % word)
        map_list.write("%d\n" % reverse_dict[word])
