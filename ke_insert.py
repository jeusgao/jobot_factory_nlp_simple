#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-29 21:37:23
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import pickle
import numpy as np
from tqdm import tqdm

import ZODB
import ZODB.FileStorage as zfs
import zc.zlibstorage
import transaction

from BTrees.Length import Length
from BTrees.LOBTree import LOBTree
from BTrees.OLBTree import OLBTree

from predictor import Predictor
from storages import DIC_ZODB, milvusDB

storage = zfs.FileStorage('hub/dbs/ke_dependency.fs')
compressed_storage = zc.zlibstorage.ZlibStorage(storage)
zoDB = ZODB.DB(compressed_storage, large_record_size=1073741824)
conn = zoDB.open()
root = conn.root()

root.dic_words = LOBTree()
root.dic_pairs = OLBTree()
root.dic_cosines = LOBTree()
root.dic_lenth = Length()


def zodb_insert(keys_cosine, keys_s, keys_o, values):
    with zoDB.transaction() as connection:
        for key_cosine, key_s, key_o, value in zip(keys_cosine, keys_s, keys_o, values):
            value['subject_id'] = key_s
            value['object_id'] = key_o

            root.dic_cosines[key_cosine] = value
            root.dic_pairs[(key_s, key_o)] = key_cosine
            root.dic_words[key_s] = value.get('subject')
            root.dic_words[key_o] = value.get('object')

    return {'msg': f'{len(keys_cosine)} records insert to milvus. {len(keys_s)+len(keys_o)} records insert to words.'}


def zodb_commit():
    transaction.commit()


def _get_rels(text, rels):
    vecs_s, vecs_o, values = [], [], []
    for r in rels:
        subj = r.get('from_word')
        pos_subj = r.get('from_pos')
        obj = r.get('to_word')
        pos_obj = r.get('to_pos')
        score = r.get('score')
        tensors = r.get('tensors')

        tensor_s = np.array(tensors.get('subject'))
        tensor_o = np.array(tensors.get('object'))

        vecs_s.append(tensor_s)
        vecs_o.append(tensor_o)
        values.append({
            'text': text,
            'subject': subj,
            'subject_pos': pos_subj,
            'object': obj,
            'object_pos': pos_obj,
            'score': score,
        })
    return vecs_s, vecs_o, values


def _get_values(ls, predictor):
    VECS_s, VECS_o, VALUES = [], [], []

    if os.path.exists('data/tmp_1.pkl'):
        print('\nLoading vecs and values ...\n')
        with open('data/tmp_1.pkl', 'rb') as f:
            _tmp = pickle.load(f)
        VECS_s = _tmp.get('vecs_s')
        VECS_o = _tmp.get('vecs_o')
        VALUES = _tmp.get('values')
    else:
        print('\nExtracting ...\n')
        for l in tqdm(ls):
            rst = predictor.predict([l.strip()], from_api=False)
            text, rels = rst.get('text'), rst.get('rels')
            vecs_s, vecs_o, values = _get_rels(text, rels)
            VECS_s += vecs_s
            VECS_o += vecs_o
            VALUES += values

        with open('data/tmp_1.pkl', 'wb') as f:
            pickle.dump({'vecs_s': VECS_s, 'vecs_o': VECS_o, 'values': VALUES}, f)

    return VECS_s, VECS_o, VALUES


def ke_insert(
    fn='sentences.txt',
    dim=1024,
    model='test',
    db_code='dependency',
    partition_tag='202103',
    index_file_size=1024,
):

    predictor = Predictor(f'{model}')

    with open(f'data/{fn}', 'r') as f:
        ls = f.read().splitlines()
    ls = set(filter(lambda x: len(x.strip()) > 1, ls))

    VECS_s, VECS_o, VALUES = _get_values(ls, predictor)

    _c = 2048
    pages = (len(VECS_s)) // _c
    if pages * _c < len(VECS_s):
        pages += 1

    root.dic_lenth.change(len(VALUES))
    print(root.dic_lenth())

    print('\nInserting ...\n')
    _maxlen = max(max(len(_s), len(_o)) for _s, _o in zip(VECS_s, VECS_o))

    for p in tqdm(range(pages)):
        _start = p * _c

        _vecs_s = VECS_s[_start:_start + _c]
        _vecs_o = VECS_o[_start:_start + _c]
        _values = VALUES[_start:_start + _c]

        _vs_s = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vecs_s]
        _vs_o = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vecs_o]

        _cosines = np.mean(np.cos(np.array(_vs_s), np.array(_vs_o)), axis=1)

        print('\nMilvus inserting ...\n')
        ids_cosine = milvusDB.insert(
            _cosines,
            dim=dim,
            collection_name=db_code,
            partition_tag=partition_tag,
            index_file_size=index_file_size,
        )

        ids_s = milvusDB.insert(
            np.mean(np.array(_vs_s), axis=1),
            dim=dim,
            collection_name='words',
            partition_tag='subjects',
            index_file_size=index_file_size,
        )
        ids_o = milvusDB.insert(
            np.mean(np.array(_vs_o), axis=1),
            dim=dim,
            collection_name='words',
            partition_tag='objects',
            index_file_size=index_file_size,
        )

        print('\nZODB inserting ...\n')
        msg = zodb_insert(ids_cosine, ids_s, ids_o, _values)
        zodb_commit()

        print(f'\n{root.dic_cosines[ids_cosine[-1]]}\n')
        print(f'\n{msg}\n')

    return 'Done.'


if __name__ == '__main__':
    rst = ke_insert()
    print(rst)
    conn.close()
    zoDB.close()
