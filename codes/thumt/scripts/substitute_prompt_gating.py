import sys
import torch
import numpy as np

TGT_INPUT=sys.argv[1]
SRC_INPUT=sys.argv[2]
SRC_ID=sys.argv[3]
OUTPUT=sys.argv[4]

name_list = [
    "gating_enc.new_adding."+SRC_ID+".0.bias",
    "gating_enc.new_adding."+SRC_ID+".0.weight",
    "gating_enc.new_adding."+SRC_ID+".2.bias",
    "gating_enc.new_adding."+SRC_ID+".2.weight",
    "gating_enc.new_adding_wte."+SRC_ID+".weight", 
    "gating_enc.new_gating."+SRC_ID+".0.bias",
    "gating_enc.new_gating."+SRC_ID+".0.weight",
    "gating_enc.new_gating."+SRC_ID+".2.bias",
    "gating_enc.new_gating."+SRC_ID+".2.weight",
    "gating_enc.new_gating_wte."+SRC_ID+".weight",
    "mbart_encoder.embed_segments.embeds."+SRC_ID+".weight",
    "prompt_enc.new_embedding."+SRC_ID+".weight",
    "prompt_enc.new_mlp_head."+SRC_ID+".0.bias",
    "prompt_enc.new_mlp_head."+SRC_ID+".0.weight",
    "prompt_enc.new_mlp_head."+SRC_ID+".2.bias",
    "prompt_enc.new_mlp_head."+SRC_ID+".2.weight"
]

src_state_dict = torch.load(SRC_INPUT, map_location="cpu")
tgt_state_dict = torch.load(TGT_INPUT, map_location="cpu")

# print(tgt_state_dict)

for n in name_list:
    tgt_state_dict['model'][n] = src_state_dict['model'][n]

torch.save(tgt_state_dict, OUTPUT)
