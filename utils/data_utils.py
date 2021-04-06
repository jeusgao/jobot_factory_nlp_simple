# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 14:51:37
# @Author  : Joe Gao (jeusgao@163.com)

import re
import json
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

p_all_sent = re.compile(r'^.*[。？\?！!；～〜…]+')


def sent_cut_zh(zh_text, cut_chars=100):
    """return full sentences at most #cut_chars Chinese characters.

    arguments:
    zh_text -- the input text string. have to be Chinese!
    cut_chars -- the maximum Chinese characters to keep (default 100)
    """
    m_all_sent = p_all_sent.match(zh_text, 0, cut_chars)
    if m_all_sent:
        return m_all_sent.group(0)
    else:
        return zh_text[:cut_chars]


def sentences_loader_train(fns=None, t1='q1', t2=None, label='label'):
    q1s, q2s, labels = [], [], []
    for fn in fns:
        sep = '\t' if fn.endswith('tsv') else ','
        df = pd.read_csv(fn, sep=sep, encoding='utf-8')
        df.dropna(inplace=True)
        q1s += list(df[t1])
        if t2:
            q2s += list(df[t2])
        labels += list(df[label])
    if not t2:
        q2s = [None] * len(q1s)

    q1s, q2s, labels = shuffle(q1s, q2s, labels, random_state=0)
    data_x = [(q1, q2) for q1, q2 in zip(q1s, q2s)]

    return data_x, labels


def sequences_loader_train(fns=None):
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
                    data_x.append([x])
                    data_y.append(y)
                x, y = [], []
            elif len(row) == 2:
                x.append(row[0])
                y.append(row[1])
    return data_x, data_y


def rel_data_loader(fns=None):
    data_x, data_y = [], []
    for fn in fns:
        with open(fn, 'r') as f:
            lines = json.load(f)
        for line in tqdm(lines, total=len(lines)):
            text = line.get('text')
            dic_rels = line.get('rels')
            dic_pairs = {}
            for k in dic_rels:
                obj_word, pos, n_rel = dic_rels.get(k).values()
                if dic_rels.get(str(n_rel)):
                    obj_pos = (pos, pos + len(obj_word))
                    subj_word, pos, _ = dic_rels.get(str(n_rel)).values()
                    subj_pos = (pos, pos + len(subj_word))

                    if subj_pos in dic_pairs:
                        dic_pairs[subj_pos].append((obj_pos, obj_word))
                    else:
                        dic_pairs[subj_pos] = [(obj_pos, obj_word)]

            data_x.append(text)
            data_y.append(dic_pairs)

    return data_x, data_y
