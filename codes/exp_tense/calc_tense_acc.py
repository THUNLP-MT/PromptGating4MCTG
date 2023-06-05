import sys

RES = sys.argv[1]
REF = sys.argv[2]
ANS = sys.argv[3]

acc_cnt, tot_cnt = 0, 0
with open(RES, 'r') as res, open(REF, 'r') as ref, open(ANS, 'w') as ans:
    for line in zip(res, ref):
        if line[0].strip() == line[1].strip():
            acc_cnt += 1
        tot_cnt += 1
    
    ans.write(str(acc_cnt) + '/' + str(tot_cnt) + '=' + str(acc_cnt/tot_cnt))

