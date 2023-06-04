import sys
import torch
import numpy as np

INPUT=sys.argv[1]
MAPLIST=sys.argv[2]
OUTPUT=sys.argv[3]

with open(MAPLIST) as fd:
  map_list = fd.readlines()
  map_list = [int(x.strip()) for x in map_list]
  map_list = map_list
  map_list = torch.Tensor(np.array(map_list), device="cpu").long()

state_dict = torch.load(INPUT, map_location="cpu")

def _shrink(state_dict, name):
  weight = state_dict[name]
  if name == 'final_logits_bias':
      weight = weight.transpose(0,1)
  shrinked_weight = torch.nn.functional.embedding(weight=weight, input=map_list)
  if name == 'final_logits_bias':
      shrinked_weight = shrinked_weight.transpose(0,1)
  state_dict[name] = shrinked_weight
  return state_dict

state_dict = _shrink(state_dict, 'model.encoder.embed_tokens.weight')
state_dict = _shrink(state_dict, 'model.decoder.embed_tokens.weight')
state_dict = _shrink(state_dict, 'model.shared.weight')
state_dict = _shrink(state_dict, 'final_logits_bias')

torch.save(state_dict, OUTPUT)
