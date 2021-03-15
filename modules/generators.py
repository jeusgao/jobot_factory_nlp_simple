#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-29 22:41:18
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import random
import numpy as np
from backend import keras


def spo_data_generator_train(
    data=None,
    y_data=None,
    tokenizer=None,
    token_dict=None,
    dim=3,
    maxlen=512,
    ML=512,
    batch_size=128,
    labeler=None,
    activation='sigmoid',
    is_sequence=False,
):
    def _get_labels(labels, pos):
        start, end = pos
        if all([start < ML, end < ML]):
            labels[start] = 1
            if end > start:
                for i in range(start + 1, end):
                    labels[i] = 2
        return labels

    while True:
        page_list = list(range((len(data) // batch_size)))
        random.shuffle(page_list)

        for page in page_list:
            start_index = page * batch_size
            end_index = start_index + batch_size

            D = data[start_index: end_index]
            Y = y_data[start_index: end_index]

            # ML = max([len(d[:maxlen]) for d in D])
            ML = maxlen

            X, LABELS_WORDS, LABELS_RELS = [], [], []
            for d, y in zip(D, Y):
                x = [token_dict.get(w, token_dict.get('[UNK]')) for w in d]
                X.append(x)

                labels_words = np.zeros(ML)
                labels_rels = np.zeros((ML, ML))
                for pos_subj in y:
                    labels_words = _get_labels(labels_words, pos_subj)
                    for pos_obj, o in y.get(pos_subj):
                        labels_words = _get_labels(labels_words, pos_obj)
                        if all([pos_obj[0] < ML, pos_subj[0] < ML]):
                            labels_rels[pos_obj[0], pos_subj[0]] = 1

                LABELS_WORDS.append(labels_words)
                LABELS_RELS.append(labels_rels)

            LABELS_WORDS = keras.utils.to_categorical(LABELS_WORDS, num_classes=3)
            LABELS_RELS = keras.utils.to_categorical(LABELS_RELS, num_classes=2)

            X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
            X_seg = np.zeros(shape=(len(X), ML))

            yield ([X, X_seg], [LABELS_WORDS, LABELS_RELS])


def data_generator_train(
    data=None,
    y_data=None,
    tokenizer=None,
    token_dict=None,
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

            ML = max([len(d[0][:maxlen]) for d in D])

            X, X_seg = [], []
            for d in D:
                if is_sequence:
                    x = [token_dict.get(w, token_dict.get('[UNK]')) for w in d[0][:ML]]
                else:
                    x, _ = tokenizer.encode(
                        first=d[0][:maxlen],
                        second=d[1][:maxlen] if len(d) == 2 and d[1] else None,
                    )
                X.append(x)

            X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
            X_seg = np.zeros(shape=(len(X), ML))

            if labeler:
                if is_sequence:
                    Y = [y[:ML] if len(y) > ML else y for y in Y]
                    Y = [[labeler.get(l) for l in y] for y in Y]
                    Y = keras.preprocessing.sequence.pad_sequences(Y, maxlen=ML, value=0, padding='post')
                else:
                    Y = np.array([labeler.get(y) for y in Y])
            else:
                Y = np.array(Y)

            if activation in ['softmax', 'crf']:
                Y = keras.utils.to_categorical(Y, num_classes=dim)

            yield ([X, X_seg], Y)


def data_generator_pred(
    data=[],
    tokenizer=None,
    token_dict=None,
    maxlen=512,
    ML=512,
    is_sequence=False,
):
    if is_sequence:
        X = [token_dict.get(w, token_dict.get('[UNK]')) for w in data[0][:ML]]
    else:
        X, _ = tokenizer.encode(
            first=data[0][:maxlen],
            second=data[1][:maxlen] if len(data) == 2 and data[1] else None,
        )

    X = keras.preprocessing.sequence.pad_sequences([X], value=0, padding='post', maxlen=ML)
    X_seg = np.zeros(shape=(len(X), ML))

    return [X, X_seg]
