# rm -r log train
set -ex
devices=7
devs=0
cuda_dir=/usr/local/cuda-11.0/lib64
code_dir=.
train_dir=/path/to/processed/yelp
vocab=/path/to/pretrain_BART/vocab.txt

exp_name=train_generation_sentiment_bs1024_negonly_yelp_bal

CUDA_VISIBLE_DEVICES=$devices \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$cuda_dir \
PYTHONPATH=$code_dir/ \
python $code_dir/thumt_gen/bin/trainer.py \
    --input $train_dir/neg_label_train.spm.txt $train_dir/neg_sent_train.spm.txt \
    --validation $train_dir/neg_label_valid.spm.txt $train_dir/neg_sent_valid.spm.txt \
    --reference $train_dir/neg_sent_valid.spm.txt \
    --output train_gen2 \
    --vocabulary $vocab $vocab \
    --model mBART --half --hparam_set big \
    --source_num 1 \
    --parameters validation_mode="loss",label_ids=[2430,1313],label_loss_weight=0.05,\
fixed_batch_size=false,batch_size=1024,train_steps=50000,update_cycle=1,device_list=[$devs],\
keep_checkpoint_max=10,save_checkpoint_steps=500,\
eval_steps=501,decode_alpha=0.6,decode_batch_size=16,keep_top_checkpoint_max=5,\
attention_dropout=0.1,relu_dropout=0.1,dropout=0.1,pattern="(new_adding.0)|(new_adding_wte.0)|(new_gating.0)|(new_gating_wte.0)|(segments.embeds.0)|(new_embedding.0)|(new_mlp_head.0)|(label_classifier)",learning_rate=1e-04,warmup_steps=5000,initial_learning_rate=5e-8,\
src_attached=[1],prompt_num=100,prompt_attached=[1],pre_encoder=false,\
src_lang_tok="en_XX",hyp_lang_tok=[],tgt_lang_tok="en_XX",\
mbart_config_path='/path/to/pretrain_BART/config.json',\
mbart_model_path='/path/to/pretrain_BART/pytorch_model.bin',\
    2>&1 | tee -a log_gen3

mv log_gen3 train_gen2
if [ ! -d ../$exp_name ]; then
    mkdir ../$exp_name
fi
mv train_gen2 ../$exp_name
