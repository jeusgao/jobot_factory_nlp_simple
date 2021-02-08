# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 14:51:37
# @Author  : Joe Gao (jeusgao@163.com)

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
            lines = f.read().splitlines()
        x, y = [], []
        for line in lines:
            row = line.strip().split()
            if len(row) == 1:
                x.append(' ')
                y.append(row[0])
            elif len(row) < 1:
                if len(x) and len(y):
                    data_x.append(x)
                    data_y.append(y)
                x, y = [], []
            elif len(row) == 2:
                x.append(row[0])
                y.append(row[1])
    return data_x, data_y
