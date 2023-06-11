#!/bin/bash

if [ -e log_gen1 ]; then
  echo "log_gen1 file already exists"
  exit 0
fi

devices=5
devs=0
cuda_dir=/usr/local/cuda-11.0/lib64
base_dir=/workspace
code_dir=.
scripts_dir=./thumt_gen/scripts
valid_dir=/path/to/test

exp_name=checkpoints
comb=1111

inp="$valid_dir/pos_label.375.spm.txt $valid_dir/asian_label.375.spm.txt $valid_dir/maskfix.pos.375.spm.txt $valid_dir/present_label.375.spm.txt $valid_dir/pre_tokens.25.spm.txt"
trg=./gen_$comb.375.spm
v_src=/path/to/pretrain_BART/vocab.txt
v_trg=/path/to/pretrain_BART/vocab.txt

ckpt=$base_dir/$exp_name/test # should have model-{1,2,...}.pt in the dir

# translate
CUDA_VISIBLE_DEVICES=$devices \
LD_LIBRARY_PATH=$cuda_dir:$LD_LIBRARY_PATH \
PYTHONPATH=$code_dir/ \
python3 $code_dir/thumt_gen/bin/translator.py \
  --input $inp --output $trg \
  --vocabulary $v_src $v_trg \
  --checkpoints $ckpt \
  --models mBART --half \
  --source_num 4 \
  --parameters beam_size=1,\
device_list=[$devs],decode_alpha=0.6,decode_batch_size=25,decode_length=100,\
src_attached=[1,1,1,1],prompt_num=100,prompt_attached=[1,1,1,1],pre_encoder=true,\
src_lang_tok="en_XX",hyp_lang_tok=["en_XX","en_XX","en_XX"],tgt_lang_tok="en_XX",\
mbart_config_path='/path/to/pretrain_BART/config.json',\
mbart_model_path='/path/to/pretrain_BART/pytorch_model.bin' \
2>&1 | tee -a log_gen1
# post-process
python $scripts_dir/spm_decode.py < $trg > $trg.dspm

if [ ! -d $ckpt/$comb ]; then
    mkdir $ckpt/$comb
fi

mv *.log $ckpt/$comb
mv log_gen1 $ckpt/$comb
mv *.spm* $ckpt/$comb
mv acc_log $ckpt/$comb

