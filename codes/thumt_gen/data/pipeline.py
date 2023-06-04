# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import torch

from thumt_gen.data.dataset import Dataset, ElementSpec, MapFunc, TextLineDataset
from thumt_gen.data.vocab import Vocabulary
from thumt_gen.tokenizers import WhiteSpaceTokenizer


def _sort_input_file(filenames, reverse=True):
    inputs = []
    input_lens = []
    files = [open(name, "rb") for name in filenames]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, [list(x) for x in zip(*sorted_inputs)]


class MTPipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        hyp_dataset = [TextLineDataset(f) for f in filenames[1:-1]] if len(filenames) > 2 else None
        tgt_dataset = TextLineDataset(filenames[-1])
        lab_dataset = TextLineDataset(filenames[-1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, [params.eos])
        hyp_dataset = [hyp.tokenize(WhiteSpaceTokenizer(), None, [params.eos]) for idp, hyp in enumerate(hyp_dataset)] if hyp_dataset is not None else None
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, [params.eos])
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        hyp_dataset = [Dataset.lookup(hyp, src_vocab, src_vocab[params.unk]) for hyp in hyp_dataset] if hyp_dataset is not None else []
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, ) + tuple(hyp_dataset) + (tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())


        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs[0], inputs[-2], inputs[-1]
            hyp_seq = inputs[1:-2] if len(inputs) > 3 else None
            src_seq = torch.tensor(src_seq)
            if hyp_seq is not None:
                hyp_seq = [torch.tensor(hyp) for hyp in hyp_seq]
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            hyp_mask = [hyp != params.vocabulary["source"][params.pad] for hyp in hyp_seq] if hyp_seq is not None else None
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            if hyp_mask is not None:
                hyp_mask = [hyp.float() for hyp in hyp_mask]
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                hyp_seq = [hyp.cuda(params.device) for hyp in hyp_seq] if hyp_seq is not None else None
                hyp_mask = [hyp.cuda(params.device) for hyp in hyp_mask] if hyp_mask is not None else None
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "hypothesis": hyp_seq,
                "hypothesis_mask": hyp_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        hyp_dataset = [TextLineDataset(f) for f in filenames[1:-1]] if len(filenames) > 2 else None
        tgt_dataset = TextLineDataset(filenames[-1])
        lab_dataset = TextLineDataset(filenames[-1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, [params.eos])
        hyp_dataset = [hyp.tokenize(WhiteSpaceTokenizer(), None, [params.eos]) for idp, hyp in enumerate(hyp_dataset)] if hyp_dataset is not None else None
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, [params.eos])
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        hyp_dataset = [Dataset.lookup(hyp, src_vocab, src_vocab[params.unk]) for hyp in hyp_dataset] if hyp_dataset is not None else []
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, ) + tuple(hyp_dataset) + (tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs[0], inputs[-2], inputs[-1]
            hyp_seq = inputs[1:-2] if len(inputs) > 3 else None
            src_seq = torch.tensor(src_seq)
            if hyp_seq is not None:
                hyp_seq = [torch.tensor(hyp) for hyp in hyp_seq]
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            hyp_mask = [hyp != params.vocabulary["source"][params.pad] for hyp in hyp_seq] if hyp_seq is not None else None
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            if hyp_mask is not None:
                hyp_mask = [hyp.float() for hyp in hyp_mask]
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                hyp_seq = [hyp.cuda(params.device) for hyp in hyp_seq] if hyp_seq is not None else None
                hyp_mask = [hyp.cuda(params.device) for hyp in hyp_mask] if hyp_mask is not None else None
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "hypothesis": hyp_seq,
                "hypothesis_mask": hyp_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_infer_dataset(filename, params, cpu=False):
        sorted_keys, sorted_data = _sort_input_file(filename)
        src_vocab = params.vocabulary["source"]

        src_dataset = TextLineDataset(sorted_data[0])
        hyp_dataset = [TextLineDataset(sorted_data[idp]) for idp in range(1, len(sorted_data)-1)] if len(sorted_data) > 2 else None
        pre_dataset = TextLineDataset(sorted_data[-1])
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, [params.eos])
        hyp_dataset = [hyp.tokenize(WhiteSpaceTokenizer(), None, [params.eos]) for idp, hyp in enumerate(hyp_dataset)] if hyp_dataset is not None else None
        pre_dataset = pre_dataset.tokenize(WhiteSpaceTokenizer(), params.bos, None)

        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        hyp_dataset = [Dataset.lookup(hyp, src_vocab, src_vocab[params.unk]) for hyp in hyp_dataset] if hyp_dataset is not None else []
        pre_dataset = Dataset.lookup(pre_dataset, src_vocab, src_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, ) + tuple(hyp_dataset) + (pre_dataset, ))

        dataset = dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())
        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, pre_seq = inputs[0], inputs[-1]
            hyp_seq = inputs[1:-1] if len(inputs) > 2 else None
            src_seq = torch.tensor(src_seq)
            if hyp_seq is not None:
                hyp_seq = [torch.tensor(hyp) for hyp in hyp_seq]
            pre_seq = torch.tensor(pre_seq)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()
            hyp_mask = [hyp != params.vocabulary["source"][params.pad] for hyp in hyp_seq] if hyp_seq is not None else None
            if hyp_mask is not None:
                hyp_mask = [hyp.float() for hyp in hyp_mask]
            pre_mask = pre_seq != params.vocabulary["source"][params.pad]
            pre_mask = pre_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                hyp_seq = [hyp.cuda(params.device) for hyp in hyp_seq] if hyp_seq is not None else None
                hyp_mask = [hyp.cuda(params.device) for hyp in hyp_mask] if hyp_mask is not None else None
                pre_seq = pre_seq.cuda(params.device)
                pre_mask = pre_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "hypothesis": hyp_seq,
                "hypothesis_mask": hyp_mask,
                "pre": pre_seq,
                "pre_mask": pre_mask
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset
