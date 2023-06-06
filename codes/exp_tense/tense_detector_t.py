import sys
import pdb
import requests
from collections import OrderedDict
import argparse
import urllib



DEFAULT_URL="http://127.0.0.1:8028/"
DEFAULT_INPUT="test.txt"
DEFAULT_OUTPUT="output.txt"

TT_UNDECIDED = 0
TT_PAST = 1
TT_PRESENT = 2
TT_NO_VERB = 3
TT_FUTURE = 4
tense_strs = {TT_UNDECIDED:"undecided",
               TT_PAST:"the past tense",
               TT_PRESENT:"the present tense",
               TT_NO_VERB:"undecided",
               TT_FUTURE:"the future tense"
}


tense_tags = {
    "VB":TT_UNDECIDED,
    "VBD":TT_PAST,
    "VBG":TT_PRESENT,
    "VBN":TT_PAST,
    "VBP":TT_PRESENT,
    "VBZ":TT_PRESENT,
    "MD":TT_FUTURE}

ORD_INDEX = 0
WORD_INDEX = 1
POS_INDEX = 2
DEP_INDEX = 3


def fetch(sent,pos_dep_url):
    r = requests.get(pos_dep_url+urllib.parse.quote(str(sent)))
    return r 



def classify_sentence(line,sorted_d):
    confidence_arr = [0]*len(tense_strs)
    score = 0
    for choice in sorted_d:
        vals = sorted_d[choice]
        if (vals["dep"] == 0):
            score = 1
        else:
            score = 1.0/vals["dep"] 
        break
    max_val = 0
    max_index = 0
    index = 0
    sum_val = 0
    curr_type = TT_UNDECIDED
    for choice in sorted_d:
        vals = sorted_d[choice]
        tmp_score = 1 if vals["dep"] == 0 else 1.0/vals["dep"]
        if (tense_tags[vals["pos"]] == TT_UNDECIDED):
            confidence_arr[TT_UNDECIDED] += tmp_score 
            if (max_val == 0):
                max_val = confidence_arr[TT_UNDECIDED]
                curr_type = TT_UNDECIDED
                max_choice = choice
        elif (tense_tags[vals["pos"]] == TT_PAST):
            confidence_arr[TT_PAST] += tmp_score
            if (confidence_arr[TT_PAST] > max_val or curr_type == TT_UNDECIDED):
                max_val = confidence_arr[TT_PAST]
                curr_type = TT_PAST
                max_choice = choice
        elif (tense_tags[vals["pos"]] == TT_FUTURE):
            confidence_arr[TT_FUTURE] += tmp_score
            if (confidence_arr[TT_FUTURE] > max_val or curr_type == TT_UNDECIDED):
                max_val = confidence_arr[TT_FUTURE]
                curr_type = TT_FUTURE
                max_choice = choice
        else:
            confidence_arr[TT_PRESENT] += tmp_score
            if (confidence_arr[TT_PRESENT] > max_val or curr_type == TT_UNDECIDED):
                max_val = confidence_arr[TT_PRESENT]
                curr_type = TT_PRESENT
                max_choice = choice
        sum_val += tmp_score
        index += 1

    if (sum_val > 0):
        confidence = max_val/sum_val
        sent_type = tense_tags[sorted_d[max_choice]["pos"]]
    else:
        sent_type = TT_NO_VERB
        confidence = 0
    #print(line,sorted_d,score,tense_strs[sent_type],round(confidence,2))
    #print(sorted_d)
    # print(str(round(score,2))+'|'+tense_strs[sent_type]+'|'+str(round(confidence,2))+'|'+line)
    return tense_strs[sent_type]

def process_file(inp_file,pos_dep_url, out_file):
    cnt = 0
    with open(inp_file) as fp, open(out_file,'w') as out:
        # print("VERB DEPTH SCORE[0-1]|tense type - undecided,present,past,future,no verb|confidence [0-1]|sentence")
        for line in fp:
            #print(line,end='')
            try:
                line = line.rstrip('\n')
                results = fetch(line,pos_dep_url)
                data = results.text.split('\n')
                verb_dict = {}
                for frag in data:
                    if  (len(frag) == 0):
                        continue
                    parts = frag.split('\t')
                    #print(parts)
                    if (parts[POS_INDEX] in tense_tags):
                        verb_dict[int(parts[ORD_INDEX])] = {"word":parts[WORD_INDEX],"pos":parts[POS_INDEX],"dep":int(parts[DEP_INDEX])}
                sorted_d = OrderedDict(sorted(verb_dict.items(), key=lambda kv: kv[1]["dep"], reverse=False))
                #print(sorted_d)
                tense_res = classify_sentence(line,sorted_d)
            except:
                tense_res = "undecided"
            out.write('The tense of this sentence is ' +tense_res+'.\n')
                
            cnt += 1
            if (cnt % 10000 == 0):
                print('done', cnt)




def main():
    parser = argparse.ArgumentParser(description='Simple tense detection using a DEP/POS tagger ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input", default=DEFAULT_INPUT,help='Input to file containing sentences.')
    parser.add_argument('-output', action="store", dest="output", default=DEFAULT_OUTPUT,help='Output file containing sentences. (default is to overwrite input file)')
    parser.add_argument('-url', action="store", dest="url", default=DEFAULT_URL,help='URL to POS/DEP server with port info. Default works with JPTDP server running on same machine')
    results = parser.parse_args()
    process_file(results.input,results.url, results.output)
    


if __name__ == "__main__":
    main()
