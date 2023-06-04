#! /usr/bin python
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob

import os
import re
import six
import time
import copy
import torch
import socket
import logging
import argparse
import numpy as np

import torch.distributed as dist

import thumt_gen.data as data
import thumt_gen.utils as utils
import thumt_gen.models as models

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score input sentences with pre-trained checkpoints.",
        usage="scorer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, nargs=2,
                        help="Path to input file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint.")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path to source and target vocabulary.")

    parser.add_argument("--source_num", type=int, default=1,
                        help="Number of sources")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")
    parser.add_argument("--half", action="store_true",
                        help="Enable Half-precision for decoding.")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        src_lang_tok="de_DE",
        hyp_lang_tok=["fr_XX"],
        tgt_lang_tok="en_XX",
        append_eos=False,
        monte_carlo=False,
        device_list=[0],
        decode_batch_size=32,
        # Dataset
        batch_size=32,
        fixed_batch_size=True,
        min_length=1,
        max_length=1024,
        buffer_size=10000,
        level="sentence",
        # prompt settings
        src_attached=[1,1],
        prompt_num=100,
        prompt_encoder_filter_size=512,
        prompt_encoder_pri_size=512,
        prompt_attached=[1,1],
        pre_encoder=False,
        only_prompt=False,
        only_prefix=False,
        label_ids=[0,1],
        label_loss_weight=0.05,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    if args.parameters:
        params.parse(args.parameters)
        params.source_num = args.source_num or params.source_num
        params.cpu = False
        # params.cpu = args.cpu or params.cpu

    params.vocabulary = {
        "source": data.Vocabulary(args.vocabulary[0]),
        "target": data.Vocabulary(args.vocabulary[1])
    }

    return params


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1

    dev_str = result.groups()[-1]
    return len(dev_str.split(","))


def main(args):
    model_cls = models.get_model(args.model)
    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)

    if args.cpu:
        dist.init_process_group("gloo", 
                                init_method=args.url,
                                rank=args.local_rank,
                                world_size=1)
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        params.device = params.device_list[args.local_rank]
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def score_fn(inputs, model, level="sentence"):
        features, labels = inputs
        loss, cnt = model(features, labels, mode="valid")
        return loss, cnt

    # Create model
    with torch.no_grad():
        if args.cpu:
            model = model_cls(params)
        else:
            model = model_cls(params).cuda()

        if args.half:
            model = model.half()

        if not params.monte_carlo:
            model.eval()

        names = glob.glob(os.path.join(args.checkpoint, "*.pt"))
        for name in names:
            if "model" in name:
                model.load_state_dict(
                    torch.load(name, map_location="cpu")["model"], strict=False)
    
                dataset = data.MTPipeline.get_train_dataset(args.input, params, cpu=args.cpu)
                data_iter = iter(dataset)
                counter = 0
                # pad_max = 1024

                # Buffers for synchronization
                size = torch.zeros([dist.get_world_size()]).long()
                l_list = [torch.empty([]).float()
                            for _ in range(dist.get_world_size())]
                c_list = [torch.empty([]).float()
                            for _ in range(dist.get_world_size())]
                loss = 0.0
                cnt = 0.0

                if dist.get_rank() == 0:
                    fd = open(args.output, "a")
                else:
                    fd = None

                while True:
                    try:
                        features = next(data_iter)
                        batch_size = features[0]["source"].shape[0]
                    except:
                        features = {
                            "source": torch.ones([1, 1]).long(),
                            "source_mask": torch.ones([1, 1]).float(),
                            "hypothesis": [torch.ones([1, 1]).long() for _ in range(params.source_num-1)] if params.source_num > 1 else None,
                            "hypothesis_mask": [torch.ones([1, 1]).float() for _ in range(params.source_num-1)] if params.source_num > 1 else None,
                            "target": torch.ones([1, 1]).long(),
                            "target_mask": torch.ones([1, 1]).float()
                        }, torch.ones([1, 1]).long()
                        batch_size = 0

                    t = time.time()
                    counter += 1

                    if args.cpu:
                        features = {
                            "source": features[0]["source"].cpu(),
                            "source_mask": features[0]["source_mask"].cpu(),
                            "hypothesis": [h.cpu() for h in features[0]["hypothesis"]] if params.source_num > 1 else None,
                            "hypothesis_mask": [h.cpu() for h in features[0]["hypothesis_mask"]] if params.source_num > 1 else None,
                            "target": features[0]["target"].cpu(),
                            "target_mask": features[0]["target_mask"].cpu()
                        }, features[1].cpu()
                        model = model.cpu()

                    _loss, _cnt = score_fn(features, model, params.level)

                    # Synchronization
                    size.zero_()
                    size[dist.get_rank()].copy_(torch.tensor(batch_size))
                    if args.cpu:
                        l_list[dist.get_rank()].copy_(_loss)
                        c_list[dist.get_rank()].copy_(_cnt)
                    else:
                        dist.all_reduce(size)
                        dist.all_gather(l_list, _loss.float())
                        dist.all_gather(c_list, _cnt.float())

                    if size.sum() == 0:
                        break

                    if dist.get_rank() != 0:
                        continue

                    for j in range(dist.get_world_size()):
                        loss += l_list[j]
                        cnt += c_list[j]

                    t = time.time() - t
                    logging.info("Finished batch: %d (%.3f sec)" % (counter, t))

                if dist.get_rank() == 0:
                    fd.write("%s\t%f\t%f\n" % (name, (loss / cnt).item(), cnt.item()))
                    fd.close()


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 8858))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    if parsed_args.cpu:
        world_size = 1
    else:
        world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
