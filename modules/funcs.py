#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-12 00:35:52
# @Author  : Joe Gao (jeusgao@163.com)

import os
from backend import tf, keras


def gather_words(inputs):
    output, tensor_words, tensor_ners = inputs
    zeros = tf.zeros((1, output.shape[-2], output.shape[-1],), dtype=tf.float32)

    words_ids = tf.argmax(tensor_words, axis=2, output_type=tf.int32)
    ners_ids = tf.argmax(tensor_ners, axis=2, output_type=tf.int32)
    _ids = tf.add(words_ids, ners_ids)
    conds = tf.expand_dims(_ids, -1)
    conds = tf.tile(conds, [1, 1, output.shape[-1]])
    tensor = tf.where(conds > 0, output, zeros)

    return tensor


def get_square(i):
    return i**4


def init_truncated_normal():
    return keras.initializers.TruncatedNormal(stddev=0.02)
