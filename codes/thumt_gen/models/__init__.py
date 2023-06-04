# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt_gen.models.transformer


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt_gen.models.transformer.Transformer
    elif name == "mbart":
        return thumt_gen.models.transformer.mBART
    else:
        raise LookupError("Unknown model %s" % name)
