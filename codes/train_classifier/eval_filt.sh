#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python /home/lzj/lzj/plug4MSG/pluginMSG/train_classifier/eval_filt.py \
    --file_loc /home/lzj/lzj/plug4MSG/train_generation_yelp_12/test/sample_pre_12/gen_sample_pre_12.375.spm.dspm \
    --specify [1,2] \
    > /home/lzj/lzj/plug4MSG/human_eval/pg/12.log
