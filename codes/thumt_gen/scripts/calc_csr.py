import sys

CONS = sys.argv[1]
RESULT = sys.argv[2]

avg_csr = 0.0

with open(CONS) as fc:
    with open(RESULT) as fr:
        fcall = fc.readlines()
        for cons in fcall:
            matched_words = 0

            cons = list(map(lambda x: x.strip(), cons.split(' ')))
            res_words = list(map(lambda x: x.strip(), fr.readline().split(' ')))
            cnt_res = dict()
            for word in res_words:
                if word in cnt_res:
                    cnt_res[word] += 1
                else:
                    cnt_res[word] = 1
            
            for word in cons:
                if word in cnt_res and cnt_res[word] > 0:
                    cnt_res[word] -= 1
                    matched_words += 1
            # print(matched_words / len(cons))
            avg_csr += matched_words / len(cons)
    print("CSR(word level):", avg_csr / len(fcall))