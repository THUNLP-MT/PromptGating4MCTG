from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fr")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fr")
model.cuda()

# translate German to French
# tokens = model.generate(**tokenizer("Dies ist ein Beispiel", return_tensors="pt"))
# results = tokenizer.batch_decode(tokens, skip_special_tokens=True)
# print(results)

SRC_FILE = sys.argv[1]
TGT_FILE = sys.argv[2]
START_LINE = int(sys.argv[3])
TOT_LINES = int(sys.argv[4])


with open(SRC_FILE, "r") as src_file, open(TGT_FILE, "w") as tgt_file:
    cnt = 0
    for lid, line in enumerate(src_file):
        if lid < START_LINE:
            continue
        elif lid >= START_LINE + TOT_LINES:
            break
        
        tokens = model.generate(**tokenizer(line, return_tensors="pt").to("cuda"))
        results = tokenizer.batch_decode(tokens.cpu(), skip_special_tokens=True)
        tgt_file.write(results[0])
        tgt_file.write("\n")
        
        cnt += 1
        if cnt % 10000 == 0:
            print("Processed {} lines".format(cnt))

print('done')
