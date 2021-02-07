#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 16:47:15
# @Author  : Joe Gao (jeusgao@163.com)

from backend import keras
from keras_bert import load_trained_model_from_checkpoint


def base_embed(base):
    return keras.Model(base.inputs, base.output)


def bert_base(
    fn_config=None,
    fn_base_model=None,
    training=False,
    trainable=True,
    seq_len=512,
):

    return load_trained_model_from_checkpoint(
        fn_config,
        fn_base_model,
        training=training,
        trainable=trainable,
        seq_len=seq_len,
    )


def get_model(inputs, output):
    return keras.Model(inputs, output)
