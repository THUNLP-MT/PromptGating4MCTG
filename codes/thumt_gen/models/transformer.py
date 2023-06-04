# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import torch
import torch.nn as nn

import thumt_gen.utils as utils
import thumt_gen.modules as modules
from thumt_gen.models.modeling_mbart import MBartForConditionalGeneration
from transformers import MBartConfig

from thumt_gen.models.PrefixTuning import PromptEncoder, GatingModule, PrefixTuning

class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, memory=None, state=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class FFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class TransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(TransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class TransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(TransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, attn_bias, encdec_bias, memory, state=None):
        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory,
                          state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, None)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class Transformer(modules.Module):

    def __init__(self, params, name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = TransformerEncoder(params)
            self.decoder = TransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return  torch.exp(-loss) * mask - (1 - mask)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def base_params_v2():
        params = Transformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = Transformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = Transformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return Transformer.base_params()
        elif name == "base_v2":
            return Transformer.base_params_v2()
        elif name == "big":
            return Transformer.big_params()
        elif name == "big_v2":
            return Transformer.big_params_v2()
        else:
            return Transformer.base_params()


class mBART(modules.Module):

    def __init__(self, params, name="mBART"):
        super(mBART, self).__init__(name=name)
        self.params = params

        if params.mbart_config_path and params.mbart_model_path:
            config = MBartConfig._dict_from_json_file(params.mbart_config_path)
            config["activation_dropout"] = params.relu_dropout
            config["attention_dropout"] = params.attention_dropout
            config["d_model"] = params.hidden_size
            config["decoder_ffn_dim"] = params.filter_size
            config["decoder_layers"] = params.num_decoder_layers
            config["dropout"] = params.dropout
            config["encoder_attention_heads"] = params.num_heads
            config["encoder_ffn_dim"] = params.filter_size
            config["encoder_layers"] = params.num_encoder_layers
            
            config = MBartConfig.from_dict(config)          
            state_dict = torch.load(params.mbart_model_path, map_location="cpu")
            mbart_model = MBartForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=None, config=config, state_dict=state_dict).model
        elif params.mbart_model_code:
            mbart_model = MBartForConditionalGeneration.from_pretrained(params.mbart_model_code).model
        else:
            raise ValueError("Unknown mbart loading scheme.")

        if params.cpu:
            mbart_model = mbart_model.cpu()

        with utils.scope(name):
            self.mbart_encoder = mbart_model.encoder
            self.mbart_decoder = mbart_model.decoder
            self.shared = self.mbart_decoder.embed_tokens
            
            if not params.only_prompt and not params.only_prefix:
                self.prompt_enc = PromptEncoder(params.hidden_size, params.prompt_encoder_filter_size, prompt_num=params.prompt_num, seg_num=params.source_num, pri_dim=params.prompt_encoder_pri_size)
                self.gating_enc = GatingModule(params.prompt_num, params.hidden_size, params.dropout, hidden_dim=params.prompt_encoder_filter_size, pri_dim=params.prompt_encoder_pri_size, seg_num=params.source_num, layer_num=params.num_encoder_layers)
        
            elif params.only_prompt:
                self.prompt_enc = PromptEncoder(params.hidden_size, params.prompt_encoder_filter_size, prompt_num=params.prompt_num, seg_num=params.source_num, pri_dim=params.prompt_encoder_pri_size)
            elif params.only_prefix:
                self.prefix_enc = PrefixTuning(params.prompt_num, params.hidden_size, params.dropout, hidden_dim=params.prompt_encoder_filter_size, pri_dim=params.prompt_encoder_pri_size, seg_num=params.source_num, layer_num=params.num_encoder_layers)
                self.prefix_dec = PrefixTuning(params.prompt_num, params.hidden_size, params.dropout, hidden_dim=params.prompt_encoder_filter_size, pri_dim=params.prompt_encoder_pri_size, seg_num=params.source_num, layer_num=params.num_decoder_layers*2)
            else:
                raise ValueError("Unknown loading scheme.")

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

        self.src_attached = params.src_attached
        self.src_attached_list = self.get_src_list(params.src_attached)
        self.prompt_attached = params.prompt_attached
        self.pre_encoder = params.pre_encoder
        
        # self.label_classifier = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size//2), nn.ReLU(), nn.Linear(params.hidden_size//2, len(params.label_ids)))
        # self.label_criterion = nn.CrossEntropyLoss()
        # self.label_loss_weight = params.label_loss_weight
        
    def get_src_list(self, base_src):
        if sum(base_src) == 0:
            raise ValueError("No source")
        
        comb_list = []
        # for i in range(len(base_src)):
        #     if base_src[i]:
        #         comb_list.append([i])
        
        # for i in range(len(comb_list)):
        #     src_att = [0 for _ in base_src]
        #     for j in comb_list[i]:
        #         src_att[j] = 1
        #     comb_list[i] = src_att
        
        for i in range(len(base_src)):
            if base_src[i]:
                src_att = [False for _ in base_src]
                src_att[i] = True
                comb_list.append(src_att)
        
        return comb_list


    def source_mask(self, origin_source_mask, mask_percentage):
        mask = torch.bernoulli(torch.ones_like(origin_source_mask, dtype=torch.float32) * (1.0-mask_percentage)).bool()
        mask = mask & origin_source_mask
        return mask


    def encode(self, features, state, mode="train", mask_percentage=0):
        # rs = random.randint(0, len(self.src_attached_list)-1)
        # self.src_attached = self.src_attached_list[rs] if mode == "train" else self.params.src_attached
        # self.prompt_attached = self.src_attached if self.params.prompt_attached == self.params.src_attached else self.params.prompt_attached
        # TODO: adapt to variable sources

        # if mode == "train":
        #     features["source_mask"] = self.source_mask(features["source_mask"], mask_percentage)

        if features["hypothesis_mask"] is not None:
            src_mask = torch.cat([features["source_mask"] if self.src_attached[0] else torch.tensor([], dtype=torch.long), ] +
                             [hyp if self.src_attached[idp+1] else torch.tensor([], dtype=torch.long) for idp, hyp in enumerate(features["hypothesis_mask"])], axis=1)
            src_seg_mask = torch.cat([torch.zeros(features["source"].shape, dtype=torch.long) if self.src_attached[0] else torch.tensor([], dtype=torch.long), ] +
                                 [torch.ones(hyp.shape, dtype=torch.long).fill_(idp+1) if self.src_attached[idp+1] else torch.tensor([], dtype=torch.long) for idp, hyp in enumerate(features["hypothesis"])], axis=1)
        else:
            src_mask = features["source_mask"]
            src_seg_mask = torch.zeros(features["source"].shape, dtype=torch.long)
        if self.params.cpu:
            src_mask = src_mask.cpu()
            src_seg_mask = src_seg_mask.cpu()
        if not self.params.only_prefix:
            src_mask = torch.cat([torch.ones([src_mask.size()[0], self.params.prompt_num * sum(self.prompt_attached)], dtype=torch.long, device=src_mask.device), src_mask], axis=1)

        encoder_output_all = self.mbart_encoder(input_ids={"src":features["source"] if self.src_attached[0] else None,
                                                    "hypo":[hyp for idp, hyp in enumerate(features["hypothesis"]) if self.src_attached[idp+1]] if features["hypothesis"] is not None else None,
                                                    "seg_mask": src_seg_mask,
                                                    "attach": self.prompt_attached}, 
                                        attention_mask=src_mask, 
                                        prompt_enc=self.prompt_enc if not self.params.only_prefix else None,
                                        gating_enc=self.gating_enc if not self.params.only_prompt and not self.params.only_prefix else None,
                                        prefix_enc=self.prefix_enc if self.params.only_prefix else None,
                                        pre_encode_fr=self.pre_encoder,
                                        fr_mask=[features["source_mask"]] + (features["hypothesis_mask"] if features["hypothesis_mask"] is not None else []))[0]
        # encoder_output_all = self.mbart_encoder(input_ids={"src":features["source"] if self.params.src_attached[0] else None, "hypo":features["hypo"] if self.params.src_attached[1] else None, "seg_mask":src_seg_mask}, attention_mask=src_mask)[0]
        state["encoder_output"] = encoder_output_all
        state["encoder_src_mask"] = src_mask


        return state

    def decode(self, features, state, mode="infer"):
        
        def _get_causal_mask(decoder_input_ids):
            def _fill_with_neg_inf(t):
                """FP16-compatible function that fills a input_ids with -inf."""
                return t.float().fill_(float("-inf")).type_as(t)
            bsz, tgt_len = decoder_input_ids.size()
            tmp = _fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
            # tmp = _fill_with_neg_inf(torch.zeros(tgt_len + self.params.suffix_token_num, tgt_len + self.params.suffix_token_num))
            mask = torch.arange(tmp.size(-1))
            tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
            # tmp = tmp * torch.cat([torch.zeros(tgt_len + self.params.suffix_token_num, self.params.suffix_token_num), torch.ones(tgt_len + self.params.suffix_token_num, tgt_len)], axis=1)
            causal_mask = tmp.to(dtype=decoder_input_ids.dtype, device=decoder_input_ids.device)
            return causal_mask

        def _invert_mask(attention_mask):
            """Turns 1->0, 0->1, False->True, True-> False"""
            assert attention_mask.dim() == 2
            return attention_mask.eq(0)

        if mode != "infer":
            tgt_mask = features["target_mask"]
            # tgt_mask = torch.cat([torch.zeros([tgt_mask.size()[0], self.params.suffix_token_num], dtype=torch.long), tgt_mask], axis=1)
            decoder_padding_mask = _invert_mask(tgt_mask)
            causal_mask = _get_causal_mask(features["target"])
            use_cache = False
            past_key_values = None
        else:
            decoder_padding_mask, causal_mask = None, None
            use_cache = True
            past_key_values = state["past_key_values"] if "past_key_values" in state \
                                else None 

        # encoder_padding_mask = torch.cat([features["source_mask"] if self.src_attached[0] else torch.tensor([], dtype=torch.long), features["hypo_mask"] if self.src_attached[1] else torch.tensor([], dtype=torch.long), features["hypo2_mask"] if self.src_attached[2] else torch.tensor([], dtype=torch.long)], axis=1)
        # encoder_padding_mask = torch.cat([torch.ones([encoder_padding_mask.size()[0], self.params.suffix_token_num * sum(self.src_attached)], dtype=torch.long), encoder_padding_mask], axis=1)
        encoder_padding_mask = state["encoder_src_mask"]

        outputs = self.mbart_decoder(input_ids=features["target"],
                                    encoder_hidden_states=state["encoder_output"],
                                    encoder_padding_mask=encoder_padding_mask,
                                    decoder_padding_mask=decoder_padding_mask,
                                    decoder_causal_mask=causal_mask,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    return_dict=True,
                                    # receiver=self.rec_prompt_tokens
                                    prefix_dec=None,#self.prefix_dec if self.params.only_prefix else None,
                                    attach=self.prompt_attached
                                    )

        if mode == "infer":
            state["past_key_values"] = outputs.past_key_values

        logits = torch.nn.functional.linear(outputs.last_hidden_state, self.shared.weight)

        if mode == "infer":
            logits = torch.squeeze(logits, axis=1)
        
        # print(features["target"], logits)
        return logits, state

    def forward(self, features, labels, mode="train", level="sentence", mask_percentage=0):
        if mode == "valid":
            mask = features["target_mask"]

            state = {}
            state = self.encode(features, state, mode="eval")
            # print(features, state)
            logits, _ = self.decode(features, state, mode="eval")
            # print(logits, labels)
            loss = self.criterion(logits, labels)
            # print(loss)
            mask = mask.to(logits)
            
            return torch.sum(loss * mask), torch.sum(mask) 
        else:
        
            # with torch.no_grad():
            #     label_target = [((features["source"] == lid).to(torch.long).sum(-1) > 0).to(torch.long) for lid in self.params.label_ids]
            #     for i_, lid in enumerate(label_target):
            #         label_target[i_][label_target[i_]>0] = i_
            #     label_target = torch.stack(label_target, dim=1).sum(1)

            # fake_tag = features["hypothesis"][-1]
            # features["hypothesis"] = features["hypothesis"][:-1]
            # features["hypothesis_mask"] = features["hypothesis_mask"][:-1]
            # # print(fake_tag.size(), fake_tag)
            # assert fake_tag.size()[1] == 2
            # fake_tag = fake_tag[:, 0].unsqueeze(1)
            # fake_tag[fake_tag==29225] = 1
            # fake_tag[fake_tag==18169] = -1
            # fake_tag = fake_tag.to(torch.float)
            # fake_tag[fake_tag<0] = -0.6
            # # print(fake_tag.size(), fake_tag)
        
            mask = features["target_mask"]

            state = {}
            state = self.encode(features, state, mode=mode, mask_percentage=mask_percentage)
            # pred_label = self.label_classifier(state["encoder_output"][:, 0, :])
            # label_loss = self.label_criterion(pred_label, label_target)
            logits, _ = self.decode(features, state, mode=mode)
            loss = self.criterion(logits, labels)
            mask = mask.to(logits)


            if mode == "eval":
                if level == "sentence":
                    return -torch.sum(loss * mask, 1)
                else:
                    return  torch.exp(-loss) * mask - (1 - mask)            

            # loss = fake_tag * loss
            # # print(loss)
            # loss[loss < -15] = -15
            # # print(loss)
            
            original_loss = (torch.sum(loss * mask) / torch.sum(mask)).to(logits)
            return original_loss # + label_loss * self.label_loss_weight


    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<s>",
            eos="</s>",
            unk="<unk>",
            hidden_size=1024,
            filter_size=4096,
            num_heads=16,
            num_encoder_layers=12,
            num_decoder_layers=12,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.1,
            label_smoothing=0.1,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            mbart_config_path="",
            mbart_model_path="",
            mbart_model_code="facebook/mbart-large-cc25",
            pre_encoder=False
        )

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return mBART.base_params()
        else:
            return mBART.base_params()
