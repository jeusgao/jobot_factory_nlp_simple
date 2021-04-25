#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-16 13:24:48
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import codecs
import numpy as np
from backend import keras
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint

class OurTokenizer(Tokenizer):

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


class EmbedModel(object):
    def __init__(self, dict_path, config_path, checkpoint_path, maxlen=32):
        self.maxlen = maxlen
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.tokenizer = OurTokenizer(self.token_dict)
        base = load_trained_model_from_checkpoint(
            config_path, checkpoint_path,
            seq_len=maxlen,
            training=False,
            trainable=False,
        )
        # output = keras.layers.Bidirectional(keras.layers.GRU(128))(base.output)
        # output = keras.layers.BatchNormalization()(output)
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
        # embs = np.mean(embs, axis=1, dtype=np.float64)

        return embs
