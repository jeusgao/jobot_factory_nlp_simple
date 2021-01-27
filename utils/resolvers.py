#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-23 19:35:02
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import numpy as np


def _get_output(word, kw_type, pos_start):
    return {
        'oriFrag': word,
        'type': kw_type,
        'beginPos': pos_start,
        'endPos': pos_start + len(word)
    }


def _get_label_sequence(id2label, preds, text):
    output = []
    word, kw_type, word, pos_start = '', '', '', 0
    for i, p in enumerate(preds):
        l = id2label.get(p, 'O')
        if len(l) > 1:
            if 'B-' in l:
                if word:
                    output.append(_get_output(word, kw_type, pos_start))
                    word = ''
                pos_start = i
                kw_type = l[2:]
                word = text[i]
            if 'I-' in l:
                kw_type = l[2:]
                word += text[i]
        else:
            if word:
                output.append(_get_output(word, kw_type, pos_start))
                word = ''

    if len(word):
        output.append(_get_output(word, kw_type, pos_start))

    return output


def _resolve_sequence(
    pred,
    text,
    id2label=None,
):
    text = sum(text, [])[0]
    print(pred[0].argmax(axis=-1).tolist()[1:len(text) + 1], text)
    _max_ner = pred[0].argmax(axis=-1).tolist()[1:len(text) + 1]
    rst_ner = _get_label_sequence(id2label, _max_ner, text)

    return rst_ner


def resolve(pred, text, activation='sigmoid', labeler=None, is_sequence=False, threshold=0.7):
    rst = None
    score = None
    if is_sequence:
        id2label = None
        if labeler:
            id2label = {v: k for k, v in labeler.items()}
        rst = _resolve_sequence(
            pred,
            text,
            id2label=id2label,
        )
    else:
        if activation == 'sigmoid':
            pred = np.asarray(pred).reshape(-1)
            rst = 0 if pred[0] < threshold else 1
            score = float(pred[0])
        else:
            rst = int(pred.argmax(-1)[0])
            score = float(np.asarray(pred).reshape(-1)[rst])
        if labeler:
            rst = labeler.get(rst, 0)

    return {'result': rst, 'score': score}
