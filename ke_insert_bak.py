#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-29 21:37:23
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import atexit
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import ZODB
import ZODB.FileStorage as zfs
import zc.zlibstorage
import transaction

from BTrees.Length import Length
from BTrees.LOBTree import LOBTree
from BTrees.OLBTree import OLBTree
from BTrees.OOBTree import OOBTree

from predictor import Predictor
from storages import DIC_ZODB, milvusDB

from utils import make_features, get_rels
from modules import EmbedModel

from storages import DIC_ZODB, milvusDB
from modules import EmbedModel

parser = argparse.ArgumentParser(description="args")
parser.add_argument("-n", "--new", type=int, default=0)
args = parser.parse_args()

collection_dependency = 'dependency'
partition_tags_cosine = 'cosine'
partition_tags_dot = 'dot'

collection_words = 'words'
partition_tags_subjects = 'subjects'
partition_tags_objects = 'objects'
zodb_code = 'dependency'

db = DIC_ZODB.get(zodb_code)
db.open()

root = db.root
root.dic_words = LOBTree()
root.dic_words_all = OLBTree()
root.dic_edges = LOBTree()
root.dic_pairs = LOBTree()
root.dic_dots = LOBTree()
root.dic_cosines = LOBTree()

fn_tmp = 'tmp_'
_C = 30000
_c = 1024

embed_model = EmbedModel(
    'hub/bases/rbtl3/vocab.txt',
    'hub/bases/rbtl3/bert_config.json',
    'hub/bases/rbtl3/bert_model.ckpt',
    maxlen=16,
)


@atexit.register
def atexit_fun():
    db.close()


def zodb_insert(keys_rel, values, ids_word, words):
    _count = 0
    with db.db.transaction() as connection:
        if len(values) > 0:
            for key_rel, value in zip(keys_rel, values):
                root.dic_edges[key_rel] = value

        if len(words) > 0:
            for id_word, word in zip(ids_word, words):
                if word not in root.dic_words_all:
                    root.dic_words[id_word] = word
                    root.dic_words_all[word] = id_word
                    _count += 1

    return {'msg': f'{len(keys_rel)} records and {_count} words embed insert to ZODB. '}

# def zodb_insert(values):
#     with zoDB.transaction() as connection:
#         for value in values:
#             _s = value.get('subject')
#             _o = value.get('object')
#             _key = f'{_s}__{_o}'
#             if _key not in root.dic_pairs:
#                 root.dic_pairs[_key] = [value]
#             else:
#                 root.dic_pairs[_key].append(value)

#     return {'msg': f'{len(values)} records insert to ZODB. '}


def zodb_commit():
    transaction.commit()


def _get_values(ls, predictor, P):
    VECS_s, VECS_o, VECS_words, VALUES, WORDS = [], [], [], [], []

    if os.path.exists(f'data/{fn_tmp}_{P}.pkl'):
        print('\nLoading vecs and values ...\n')
        with open(f'data/{fn_tmp}_{P}.pkl', 'rb') as f:
            _tmp = pickle.load(f)
        VECS_s = _tmp.get('VECS_s')
        VECS_o = _tmp.get('VECS_o')
        VALUES = _tmp.get('VALUES')
        VECS_words = _tmp.get('VECS_words')
        WORDS = _tmp.get('WORDS')
    else:
        print(f'\nExtracting page - {P}...\n')
        for l in tqdm(ls, desc=f'Page - {P}'):
            rst = predictor.predict([l.strip()], from_api=False)
            text, rels = rst.get('text'), rst.get('rels')
            if len(rels) > 0:
                vecs_s, vecs_o, values = get_rels(text, rels)
                VECS_s += vecs_s
                VECS_o += vecs_o
                VALUES += values

                _words_s = [v.get('subject') for v in values if v.get('subject') not in root.dic_words_all]
                _words_o = [v.get('object') for v in values if v.get('object') not in root.dic_words_all]
                _words = list(set(_words_s + _words_o))
                WORDS += _words

        if len(WORDS) > 0:
            WORDS = list(set(WORDS))
            VECS_words = embed_model.get_embed(WORDS)

        print(f'\nDumping data/{fn_tmp}_{P}.pkl...\n')
        with open(f'data/{fn_tmp}_{P}.pkl', 'wb') as f:
            pickle.dump({
                'VECS_s': VECS_s,
                'VECS_o': VECS_o,
                'VALUES': VALUES,
                'VECS_words': VECS_words,
                'WORDS': WORDS,
            }, f)

    return VECS_s, VECS_o, VALUES, VECS_words, WORDS


def ke_insert(
    fn='sentences_dependency.txt',
    dim=1024,
    model='test',
    db_code='dependency',
    partition_tag='dot',
    index_file_size=1024,
    new=False,
):
    if new:
        milvusDB.drop_collection(db_code)
        milvusDB.drop_collection('words')

    predictor = Predictor(f'{model}')

    with open(f'data/{fn}', 'r') as f:
        ls = f.read().splitlines()
    ls = list(set(filter(lambda x: len(x.strip()) > 4, ls)))
    print(len(ls))

    PAGES = len(ls) // _C
    if PAGES * _C < len(ls):
        PAGES += 1

    for P in tqdm(range(PAGES)):
        _start = P * _C
        VECS_s, VECS_o, VALUES, VECS_words, WORDS = _get_values(ls[_start:_start + _C], predictor, P)
        VECS_words = np.mean(VECS_words, axis=1)
        print('LENTHs:',len(WORDS), len(VALUES))

        if all([len(WORDS), len(VECS_words), len(VECS_s), len(VECS_o), len(VALUES)]):
            _MAX = max(len(WORDS), len(VALUES))
            pages = _MAX // _c
            if pages * _c < _MAX:
                pages += 1

            print('\nInserting ...\n')
            _maxlen = max(max(len(_s), len(_o)) for _s, _o in zip(VECS_s, VECS_o))

            for p in tqdm(range(pages)):
                _start = p * _c

                _vecs_words = VECS_words[_start:_start + _c]
                _words = WORDS[_start:_start + _c]

                _vecs_s = VECS_s[_start:_start + _c]
                _vecs_o = VECS_o[_start:_start + _c]
                _values = VALUES[_start:_start + _c]

                print('\nMilvus inserting ...\n')
                ids_dot = None
                if len(_vecs_s) > 0:
                    relation, _vs_s, _vs_o = make_features(_maxlen, _vecs_s, _vecs_o)
                    ids_dot = milvusDB.insert(
                        relation,
                        dim=dim,
                        collection_name=db_code,
                        partition_tag=partition_tag,
                        index_file_size=index_file_size,
                        metric='IP',
                    )

                ids_word = None
                if len(_vecs_words) > 0:
                    ids_word = milvusDB.insert(
                        _vecs_words,
                        dim=1024,
                        collection_name='words',
                        index_file_size=index_file_size,
                        metric='L2',
                    )
                print(f'\n{len(_vecs_s)} records and {len(_words)} words embed insert to milvus.\n')

                print('\nZODB inserting ...\n')
                msg = zodb_insert(ids_dot, _values, ids_word, _words)
                zodb_commit()

                print(f'\n{msg}\n')

    return 'Done.'


if __name__ == '__main__':
    rst = ke_insert(new=args.new == 1)
    print(rst)
    db.close()
