#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-29 22:41:18
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import random
import numpy as np
from tensorflow import keras


def data_generator_train(
    data=None,
    y_data=None,
    tokenizer=None,
    dim=2,
    maxlen=512,
    ML=512,
    batch_size=128,
    labeler=None,
    activation='sigmoid',
    is_sequence=False,
):
    while True:
        page_list = list(range((len(data) // batch_size)))
        random.shuffle(page_list)

        for page in page_list:
            start_index = page * batch_size
            end_index = start_index + batch_size

            D = data[start_index: end_index]
            Y = y_data[start_index: end_index]

            X, X_seg = [], []
            for d in D:
                x, x_seg = tokenizer.encode(
                    first=d[0][:maxlen],
                    second=d[1][:maxlen] if len(d) == 2 else None,
                )
                X.append(x)
                X_seg.append(x_seg)

            X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
            X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=ML)

            if labeler and is_sequence:
                Y = [['O'] + y[:ML - 1] + ['O'] if len(y) >= ML - 2 else y for y in Y]
                Y = [['O'] + y + ['O'] * (ML - 1 - len(y)) if len(y) < ML - 2 else y for y in Y]
                Y = [[labeler.get(l) for l in y] for y in Y]
                Y = keras.preprocessing.sequence.pad_sequences(Y, maxlen=ML, value=0, padding='post')
            else:
                Y = np.array(Y)

            if activation in ['softmax', 'crf']:
                Y = keras.utils.to_categorical(Y, num_classes=dim)

            yield ([X, X_seg], Y)


def data_generator_pred(
    data=[],
    tokenizer=None,
    maxlen=512,
    ML=512,
):
    X, X_seg = tokenizer.encode(
        first=data[0][:maxlen],
        second=data[1][:maxlen] if len(data) == 2 else None,
    )

    X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
    X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=ML)

    return [X, X_seg]
