#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-29 21:37:23
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import numpy as np
from tqdm import tqdm

import ZODB
import ZODB.FileStorage as zfs
import zc.zlibstorage
import transaction

from predictor import Predictor
from storages import DIC_ZODB, milvusDB

storage = zfs.FileStorage('hub/dbs/ke_dependency.fs')
compressed_storage = zc.zlibstorage.ZlibStorage(storage)
zoDB = ZODB.DB(compressed_storage, large_record_size=536870912)
conn = zoDB.open()
root = conn.root()


def zodb_insert(keys, values):
    with zoDB.transaction() as connection:
        for key, value in zip(keys, values):
            root[key] = value
    # transaction.commit()
    return {'msg': f'record {keys}: {values} insert successfully.'}


def zodb_commit():
    transaction.commit()


def _get_rels(text, rels):
    vecs, values = [], []
    for r in rels:
        subj = r.get('from_word')
        pos_subj = r.get('from_pos')
        obj = r.get('to_word')
        pos_obj = r.get('to_pos')
        score = r.get('score')
        tensors = r.get('tensors')
        tensor_s = np.array(tensors.get('subject'))
        tensor_o = np.array(tensors.get('object'))
        vec = np.concatenate((tensor_s, tensor_o))

        vecs.append(vec.tolist())
        values.append({
            'text': text,
            'subject': subj,
            'subject_pos': pos_subj,
            'object': obj,
            'object_pos': pos_obj,
            'score': score,
        })
    return vecs, values


def ke_insert(
    fn='sentences_dependency.txt',
    dim=2048,
    model='test',
    db_code='dependency',
    partition_tag='202103',
    index_file_size=1024,
):

    predictor = Predictor(f'{model}')

    with open(f'data/{fn}', 'r') as f:
        ls = f.read().splitlines()
    ls = set(filter(lambda x: len(x.strip()) > 1, ls))

    VECS, VALUES = [], []

    if os.path.exists('data/tmp.pkl'):
        print('\nLoading vecs and values ...\n')
        with open('data/tmp.pkl', 'rb') as f:
            _tmp = pickle.load(f)
        VECS = _tmp.get('vecs')
        VALUES = _tmp.get('values')
    else:
        print('\nExtracting ...\n')

        for l in tqdm(ls):
            rst = predictor.predict([l.strip()])
            text, rels = rst.get('text'), rst.get('rels')
            vecs, values = _get_rels(text, rels)
            VECS += vecs
            VALUES += values

        import pickle

        with open('data/tmp.pkl', 'wb') as f:
            pickle.dump({'vecs': VECS, 'values': VALUES}, f)

    _c = 1024
    pages = (len(VECS)) // _c
    if pages * _c < len(VECS):
        pages += 1

    print('\nInserting ...\n')

    for p in tqdm(range(pages)):
        _start = p * _c
        _vecs = VECS[_start:_start + _c]
        _values = VALUES[_start:_start + _c]

        ids = milvusDB.insert(
            _vecs,
            dim=dim,
            collection_name=db_code,
            partition_tag=partition_tag,
            index_file_size=index_file_size,
        )

        msg = zodb_insert(ids, _values)
        zodb_commit()

    return 'Done.'


if __name__ == '__main__':
    rst = ke_insert()
    print(rst)
    conn.close()
    zoDB.close()
