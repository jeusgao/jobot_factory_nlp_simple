#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-16 13:24:48
# @Author  : Joe Gao (jeusgao@163.com)

import os
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint

from backend import keras
from modules import tokenizer_zh


class EmbedModel(object):
    def __init__(self, dict_path, config_path, checkpoint_path, maxlen=32):
        self.maxlen = maxlen
        self.token_dict, self.tokenizer = tokenizer_zh(fn_vocab=dict_path)
        base = load_trained_model_from_checkpoint(
            config_path, checkpoint_path,
            seq_len=maxlen,
            training=False,
            trainable=False,
        )
        self.model = keras.models.Model(base.input, base.output)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    def get_embed(self, texts):
        X, X_seg = [], []
        for t in texts:
            x, x_seg = self.tokenizer.encode(t[:self.maxlen - 2])
            X.append(x)
            X_seg.append(x_seg)

        X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=self.maxlen)
        X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=self.maxlen)

        embs = self.model.predict([X, X_seg])

        return embs
