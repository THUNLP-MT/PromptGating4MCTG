import sys
import torch
import numpy as np

TGT_INPUT=sys.argv[1]
SRC_INPUT=sys.argv[2]
SRC_ID=sys.argv[3]
TGT_ID=sys.argv[4]
OUTPUT=sys.argv[5]

src_name_list = [
    "mbart_encoder.embed_segments.embeds."+SRC_ID+".weight",
    "prompt_enc.new_embedding."+SRC_ID+".weight",
    "prompt_enc.new_mlp_head."+SRC_ID+".0.bias",
    "prompt_enc.new_mlp_head."+SRC_ID+".0.weight",
    "prompt_enc.new_mlp_head."+SRC_ID+".2.bias",
    "prompt_enc.new_mlp_head."+SRC_ID+".2.weight"
]
tgt_name_list = [
    "mbart_encoder.embed_segments.embeds."+TGT_ID+".weight",
    "prompt_enc.new_embedding."+TGT_ID+".weight",
    "prompt_enc.new_mlp_head."+TGT_ID+".0.bias",
    "prompt_enc.new_mlp_head."+TGT_ID+".0.weight",
    "prompt_enc.new_mlp_head."+TGT_ID+".2.bias",
    "prompt_enc.new_mlp_head."+TGT_ID+".2.weight"
]

src_state_dict = torch.load(SRC_INPUT, map_location="cpu")
tgt_state_dict = torch.load(TGT_INPUT, map_location="cpu")

# print(tgt_state_dict)

for n1, n2 in zip(src_name_list, tgt_name_list):
    tgt_state_dict['model'][n2] = src_state_dict['model'][n1]

torch.save(tgt_state_dict, OUTPUT)
