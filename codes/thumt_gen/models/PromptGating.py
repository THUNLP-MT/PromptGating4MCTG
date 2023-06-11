from functools import reduce
import math
from torch import nn, arange, cat, Tensor
import torch.nn.functional as F
import torch

class SegmentEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # self.embeds = nn.Embedding(num_embeddings, embedding_dim)
        self.embeds = nn.ModuleList([nn.Embedding(1, embedding_dim) for _ in range(num_embeddings)])

    def forward(self, input_ids):
        # seg_embeds = self.embeds(input_ids)
        seg_embeds = self.embeds[input_ids[0, 0].item()](torch.zeros_like(input_ids, dtype=torch.long))
        return seg_embeds

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, prompt_num=100, seg_num=2, pri_dim=512):
        super().__init__()
        self.embedding_size = embed_dim
        self.hidden_size = hidden_dim
        self.pri_size = pri_dim
        self.prompt_num = prompt_num
        self.seg_num = seg_num
        self.generate_parameters(prompt_num)

    def generate_parameters(self, prompt_num) -> None:
        self.new_embedding = nn.ModuleList([nn.Embedding(prompt_num, self.pri_size) for _ in range(self.seg_num)])
        self.new_ids = nn.Parameter(arange(self.prompt_num, dtype=torch.long, requires_grad=False), requires_grad = False)
        self.new_mlp_head = nn.ModuleList([nn.Sequential(
            nn.Linear(self.pri_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embedding_size)
        ) for _ in range(self.seg_num)])

    def forward(self, attach):
        if attach is None or not reduce(lambda x, y: x or y, attach[0:self.seg_num]) or self.prompt_num == 0:
            return None

        prompt_embeds = [self.new_embedding[id](self.new_ids).unsqueeze(0) if attach[id] else None for id in range(self.seg_num)]
        prompt_embeds = cat([self.new_mlp_head[id](prompt_embeds[id]) if attach[id] else Tensor([]) for id in range(self.seg_num)], axis=1)

        return prompt_embeds

class GatingModule(nn.Module):
    def __init__(self, prompt_num: int, embed_dim: int, dropout, hidden_dim: int = 512, pri_dim: int = 512, seg_num: int = 2, layer_num: int = 12) -> None:
        super().__init__()
        self.prompt_num = prompt_num
        self.seg_num = seg_num
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.new_adding_gating_ids = nn.Parameter(arange(self.prompt_num, dtype=torch.long, requires_grad=False), requires_grad = False)
        self.new_adding_wte = nn.ModuleList([nn.Embedding(prompt_num, pri_dim) for _ in range(self.seg_num)])
        self.new_adding = nn.ModuleList([nn.Sequential(
            nn.Linear(pri_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim*layer_num)) for _ in range(self.seg_num)])
        
        self.new_gating_wte = nn.ModuleList([nn.Embedding(prompt_num, pri_dim) for _ in range(self.seg_num)])
        self.new_gating = nn.ModuleList([nn.Sequential(
            nn.Linear(pri_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim*layer_num),#) for _ in range(self.seg_num)])
            nn.Sigmoid()) for _ in range(self.seg_num)])

        self.dropout = dropout
    
    def forward(self, attach=None):
        if attach is None or not reduce(lambda x, y: x or y, attach[0:self.seg_num]) or self.prompt_num == 0:
            return None

        adding_embeds = [self.new_adding_wte[i](self.new_adding_gating_ids) if attach[i] else None for i in range(self.seg_num)]
        adding_embeds = [self.new_adding[i](adding_embeds[i]) if attach[i] else None for i in range(self.seg_num)]
        adding_embeds = cat([F.dropout(adding_embeds[i], p=self.dropout, training=self.training) if attach[i] else Tensor([]) for i in range(self.seg_num)], axis=0)
        adding_embeds = adding_embeds.view(-1, self.layer_num, self.embed_dim)
                
        # gate
        gating_embeds = [self.new_gating_wte[i](self.new_adding_gating_ids) if attach[i] else None for i in range(self.seg_num)]
        gating_embeds = [self.new_gating[i](gating_embeds[i]) if attach[i] else None for i in range(self.seg_num)]
        gating_embeds = cat([F.dropout(gating_embeds[i], p=self.dropout, training=self.training) if attach[i] else Tensor([]) for i in range(self.seg_num)], axis=0)
        gating_embeds = gating_embeds.view(-1, self.layer_num, self.embed_dim)
        
        return adding_embeds, gating_embeds


class PrefixTuning(nn.Module):
    def __init__(self, prompt_num: int, embed_dim: int, dropout, hidden_dim: int = 512, pri_dim: int = 512, seg_num: int = 2, layer_num: int = 12) -> None:
        super().__init__()
        self.prompt_num = prompt_num
        self.seg_num = seg_num
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.prefix_tokens = nn.Parameter(arange(self.prompt_num, dtype=torch.long, requires_grad=False), requires_grad = False)
        self.prefix_wte = nn.ModuleList([nn.Embedding(prompt_num, pri_dim) for _ in range(self.seg_num)])
        self.prefix_enc = nn.ModuleList([nn.Sequential(
            nn.Linear(pri_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, 2*embed_dim*layer_num)) for _ in range(self.seg_num)])

        self.dropout = dropout
    
    def forward(self, attach=None):
        if attach is None or not reduce(lambda x, y: x or y, attach[0:self.seg_num]) or self.prompt_num == 0:
            return None

        key_value = [self.prefix_wte[i](self.prefix_tokens) if attach[i] else None for i in range(self.seg_num)]
        key_value = [self.prefix_enc[i](key_value[i]) if attach[i] else None for i in range(self.seg_num)]
        key_value = cat([F.dropout(key_value[i], p=self.dropout, training=self.training) if attach[i] else Tensor([]) for i in range(self.seg_num)], axis=0)
        key_value = key_value.view(-1, self.layer_num, self.embed_dim*2)
        key_value = torch.split(key_value, self.embed_dim, dim=2)


        return key_value[0], key_value[1]

class AdapterLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim=24, seg_num=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seg_num = seg_num
        self.instantiate()

    def instantiate(self):
        self.modulelist = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.input_dim)) for _ in range(self.seg_num)])

    def forward(self, output, attach=None):
        if not isinstance(output, torch.Tensor):
            raise TypeError
        
        if attach is None or not reduce(lambda x, y: x or y, attach[0:self.seg_num]):
            return None
        
        ret = None
        for ida, a in enumerate(attach[0:self.seg_num]):
            if a:
                if ret is None:
                    ret = self.modulelist[ida](output)
                else:
                    ret = self.modulelist[ida](ret)
        
        return ret
