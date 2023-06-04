# rm -r log train
set -ex
devices=6
devs=0
cuda_dir=/usr/local/cuda-11.0/lib64
base_dir=/home/lzj/lzj/plug4MSG/pluginMSG
code_dir=.
train_dir=/home/lzj/lzj/plug4MSG/data/wmt16ende
valid_dir=/home/lzj/lzj/plug4MSG/data/wmt16ende
vocab=/home/lzj/lzj/plug4MSG/data/vocab.endefr_6w.fixed.txt
ckpt_pt=/home/lzj/lzj/plug4MSG/train_fr_enc_sing/test/model-5.pt

exp_name=train_fr_enc

CUDA_VISIBLE_DEVICES=$devices \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$cuda_dir \
PYTHONPATH=$code_dir/ \
python $code_dir/thumt/bin/trainer.py \
    --input $train_dir/train.spm.de $train_dir/train.spm.en.masked.num.s $train_dir/train.spm.en.clause $train_dir/train.spm.fr $train_dir/train.spm.en \
    --output train1 \
    --vocabulary $vocab $vocab \
    --validation $valid_dir/newstest2013.spm.de $valid_dir/newstest2013.spm.en.masked.num.s $valid_dir/newstest2013.spm.en.clause $valid_dir/newstest2013.spm.fr.fromen \
    --references $valid_dir/newstest2013.spm.en \
    --model mBART --half --hparam_set big \
    --source_num 4 \
    --checkpoint $ckpt_pt \
    --parameters \
fixed_batch_size=false,batch_size=512,train_steps=150000,update_cycle=1,device_list=[$devs],\
keep_checkpoint_max=10,save_checkpoint_steps=5000,\
eval_steps=5001,decode_alpha=0.6,decode_batch_size=16,keep_top_checkpoint_max=5,\
attention_dropout=0.1,relu_dropout=0.1,dropout=0.1,pattern="(new_adding.3)|(new_adding_wte.3)|(new_gating.3)|(new_gating_wte.3)|(segments.embeds.3)|(new_embedding.3)|(new_mlp_head.3)|(fr_encoder)",learning_rate=1e-05,warmup_steps=10000,initial_learning_rate=5e-8,\
src_attached=[1,0,0,1],prompt_num=100,prompt_attached=[1,0,0,1],pre_encoder=true,\
src_lang_tok="de_DE",hyp_lang_tok=["en_XX","en_XX","fr_XX"],tgt_lang_tok="en_XX",\
mbart_config_path='/home/lzj/lzj/plug4MSG/data/config.endefr_6w.json',\
mbart_model_path='/home/lzj/lzj/plug4MSG/data/pytorch_model.endefr_6w.bin',\
spm_path='/home/lzj/lzj/prompt4MSG/pretrain/sentence.bpe.model'\
    2>&1 | tee -a log2

mv log2 train1
if [ ! -d ../$exp_name ]; then
    mkdir ../$exp_name
fi
mv train1 ../$exp_name
