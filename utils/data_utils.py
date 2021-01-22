# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 14:51:37
# @Author  : Joe Gao (jeusgao@163.com)

from tensorflow import keras

import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def sentences_loader_train(fns=None, dir_data=None, data_cls=None, t1='q1', t2=None, label='label'):
    q1s, q2s, labels = [], [], []
    for fn in fns:
        sep = '\t' if fn.endswith('tsv') else ','
        df = pd.read_csv(fn, sep=sep, encoding='utf-8')
        df.dropna(inplace=True)
        q1s += list(df[t1])
        if t2:
            q2s += list(df[t2])
        labels += list(df[label])

    q1s, q2s, labels = shuffle(q1s, q2s, labels, random_state=0)
    data_x = [(q1, q2 if t2 else None) for q1, q2 in zip(q1s, q2s)]

    return data_x, labels


def sequences_loader_train(fns=None, dir_data=None, data_cls=None):
    data_x, data_y = [], []
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        x, y = [], []
        for line in lines:
            row = line.split()
            if len(row) <= 1:
                data_x.append(x)
                data_y.append(y)
                x = []
                y = []
            else:
                x.append(row[0])
                y.append(row[1])
    return data_x, data_y


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
                    second=d[1][:maxlen] if d[1] else None,
                )
                X.append(x)
                X_seg.append(x_seg)

            X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
            X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=ML)

            if labeler:
                Y = [['O'] + y[:ML - 1] if len(y) > ML - 1 else y for y in Y]
                Y = [['O'] + y + ['O'] * (ML - 1 - len(y)) if len(y) < ML - 1 else y for y in Y]
                Y = [[labeler.get(l) for l in y] for y in Y]
                Y = keras.preprocessing.sequence.pad_sequences(Y, maxlen=ML, value=0, padding='post')
            else:
                Y = np.array(Y)

            if activation in ['softmax', 'crf']:
                Y = keras.utils.to_categorical(Y, num_classes=dim)

            yield ([X, X_seg], Y)


def data_generator_eval(
    X=None,
    Y=None,
    tokenizer=None,
    dim=2,
    maxlen=512,
    ML=512,
    batch_size=128,
    labeler=None,
    activation='sigmoid',
):
    X, X_seg = [], []
    for d in tqdm.tqdm(X):
        x, x_seg = tokenizer.encode(
            first=d[0][:maxlen],
            second=d[1][:maxlen] if d[1] else None,
        )
        X.append(x)
        X_seg.append(x_seg)

    X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
    X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=ML)

    if labeler:
        Y = [['O'] + y[:ML - 1] if len(y) > ML - 1 else y for y in Y]
        Y = [['O'] + y + ['O'] * (ML - 1 - len(y)) if len(y) < ML - 1 else y for y in Y]
        Y = [[labeler.get(l) for l in y] for y in Y]
        Y = keras.preprocessing.sequence.pad_sequences(Y, maxlen=ML, value=0, padding='post')
    else:
        Y = np.array(Y)

    if activation in ['softmax', 'crf']:
        Y = keras.utils.to_categorical(Y, num_classes=dim)

    return [X, X_seg], Y


def data_generator_pred(
    data=None,
    tokenizer=None,
    maxlen=512,
    ML=512,
):
    X, X_seg = tokenizer.encode(
        first=data[0][:maxlen],
        second=data[1][:maxlen] if data[1] else None,
    )

    X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=ML)
    X_seg = keras.preprocessing.sequence.pad_sequences(X_seg, value=0, padding='post', maxlen=ML)

    return [X, X_seg]
