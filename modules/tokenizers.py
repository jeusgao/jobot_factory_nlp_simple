#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 12:31:27
# @Author  : Joe Gao (jeusgao@163.com)

import codecs
from keras_bert import Tokenizer


class TokenizerZh(Tokenizer):
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


def tokenizer_zh(fn_vocab=None):
    token_dict = {}
    with codecs.open(fn_vocab, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = TokenizerZh(token_dict)

    return token_dict, tokenizer


def kwr_labeler(labeler=None, y_data=None):
    if y_data:
        for seq in y_data:
            for y in seq:
                if y not in labeler:
                    labeler[y] = len(labeler)

    return labeler
