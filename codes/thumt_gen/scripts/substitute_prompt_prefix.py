import sys
import torch
import numpy as np

TGT_INPUT=sys.argv[1]
SRC_INPUT=sys.argv[2]
SRC_ID=sys.argv[3]
TGT_ID=sys.argv[4]
OUTPUT=sys.argv[5]

src_name_list = [
    "prefix_enc.prefix_enc."+SRC_ID+".0.bias",
    "prefix_enc.prefix_enc."+SRC_ID+".0.weight",
    "prefix_enc.prefix_enc."+SRC_ID+".2.bias",
    "prefix_enc.prefix_enc."+SRC_ID+".2.weight",
    "prefix_enc.prefix_wte."+SRC_ID+".weight", 
    "mbart_encoder.embed_segments.embeds."+SRC_ID+".weight",
]
tgt_name_list = [
    "prefix_enc.prefix_enc."+TGT_ID+".0.bias",
    "prefix_enc.prefix_enc."+TGT_ID+".0.weight",
    "prefix_enc.prefix_enc."+TGT_ID+".2.bias",
    "prefix_enc.prefix_enc."+TGT_ID+".2.weight",
    "prefix_enc.prefix_wte."+TGT_ID+".weight", 
    "mbart_encoder.embed_segments.embeds."+TGT_ID+".weight",
]

src_state_dict = torch.load(SRC_INPUT, map_location="cpu")
tgt_state_dict = torch.load(TGT_INPUT, map_location="cpu")

# print(tgt_state_dict)

for n1, n2 in zip(src_name_list, tgt_name_list):
    tgt_state_dict['model'][n2] = src_state_dict['model'][n1]

torch.save(tgt_state_dict, OUTPUT)
