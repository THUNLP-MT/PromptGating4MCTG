#!/bin/bash

if [ -e log1 ]; then
  echo "log1 file already exists"
  exit 0
fi

devices=0
devs=0
cuda_dir=/usr/local/cuda-11.0/lib64
base_dir=/workspace
code_dir=.
scripts_dir=./thumt/scripts
train_dir=/path/to/processed/wmt
valid_dir=/path/to/processed/wmt
comb=1111

exp_name=train_mask_tense_fr

for set in newstest2014; do

inp="$valid_dir/$set.spm.de $valid_dir/$set.spm.en.masked.num.s $valid_dir/$set.spm.en.tense $valid_dir/$set.spm.fr"
ref=$valid_dir/$set.en
ref_csr=/path/to/processed/wmt/newstest2014.en.masked.num.s

trg=./$set.trans.spm
v_src=/path/to/pretrain_mBART/vocab.endefr_6w.fixed.txt
v_trg=/path/to/pretrain_mBART/vocab.endefr_6w.fixed.txt

ckpt=$base_dir/$exp_name/test # should have model-{1,2,...}.pt in the dir

echo "translating $set"

# translate
CUDA_VISIBLE_DEVICES=$devices \
LD_LIBRARY_PATH=$cuda_dir:$LD_LIBRARY_PATH \
PYTHONPATH=$code_dir/ \
python3 $code_dir/thumt/bin/translator.py \
  --input $inp --output $trg \
  --vocabulary $v_src $v_trg \
  --checkpoints $ckpt \
  --models mBART --half \
  --source_num 4 \
  --parameters device_list=[$devs],decode_alpha=1.2,decode_batch_size=32,decode_length=50,\
src_attached=[1,1,1,1],prompt_num=100,prompt_attached=[1,1,1,1],pre_encoder=true,\
src_lang_tok="de_DE",hyp_lang_tok=["en_XX","en_XX","fr_XX"],tgt_lang_tok="en_XX",\
mbart_config_path='/path/to/pretrain_mBART/config.endefr_6w.json',\
mbart_model_path='/path/to/pretrain_mBART/pytorch_model.endefr_6w.bin' \
2>&1 | tee -a log1
# post-process
python $scripts_dir/spm_decode.py < $trg > $trg.dspm
# multi-bleu-detok
sacrebleu $ref < $trg.dspm | tee ./bleu_$set.log
#perl $scripts_dir/multi-bleu-detok.perl $ref < ${trg}.dspm | tee ./bleu_$set.log
# calculate CSR
python thumt/scripts/calc_csr.py $ref_csr $trg.dspm >> ./bleu_$set.log

if [ ! -d $ckpt/$comb ]; then
    mkdir $ckpt/$comb
fi

mv *.log $ckpt/$comb
mv log1 $ckpt/$comb
mv *.spm* $ckpt/$comb


done
