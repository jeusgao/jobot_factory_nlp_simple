#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-29 21:37:23
# @Author  : Joe Gao (jeusgao@163.com)

import os
import time
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
root.dic_vocab = OLBTree()
root.dic_words = LOBTree()
root.dic_edges = LOBTree()
root.dic_word_edges = LOBTree()
root.dic_pairs = OOBTree()
root.dic_word_pairs = OOBTree()

fn_tmp = 'tmp/tmp_'
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


def zodb_insert_words(keys, words):
    with db.db.transaction() as connection:
        if len(words) > 0:
            for key, word in zip(keys, words):
                root.dic_words[key] = word
                root.dic_vocab[word] = key
    return f'{len(root.dic_vocab)}, {len(root.dic_words)} words insert to ZODB. '


def zodb_insert_rels(keys_rel, values):
    with db.db.transaction() as connection:
        if len(values) > 0:
            for key_rel, value in zip(keys_rel, values):
                _s, _o = value.get('subject'), value.get('object')
                key_s, key_o = root.dic_vocab[_s], root.dic_vocab[_o]

                root.dic_edges[key_rel] = {
                    'subject': _s,
                    'subject_id': key_s,
                    'object': _o,
                    'object_id': key_o
                }

                if key_s in root.dic_word_edges:
                    root.dic_word_edges[key_s].append(key_rel)
                else:
                    root.dic_word_edges[key_s] = [key_rel]

                if key_o in root.dic_word_edges:
                    root.dic_word_edges[key_o].append(key_rel)
                else:
                    root.dic_word_edges[key_o] = [key_rel]

                root.dic_pairs[(key_s, key_o)] = value
                root.dic_word_pairs[(_s, _o)] = (key_s, key_o)

    return f'{len(keys_rel)} edges insert to ZODB. '


def zodb_commit():
    transaction.commit()


def _get_values(ls, predictor, P):
    VECS_s, VECS_o, VALUES, VECS_words, WORDS = [], [], [], [], []

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

                _words_s = [v.get('subject') for v in values if v.get('subject') not in root.dic_vocab]
                _words_o = [v.get('object') for v in values if v.get('object') not in root.dic_vocab]
                WORDS += _words_s + _words_o
        WORDS = list(set(WORDS))

        print(f'WORDS Counts: {len(WORDS)}.')

        s = time.time()
        VECS_words += embed_model.get_embed(WORDS).tolist()
        print(f'\n{len(WORDS)} words embeded costed {time.time() - s}.\n')

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
    index_file_size=2048,
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
        print('LENTHs: Words -', len(WORDS), np.array(VECS_words).shape, 'Pairs', len(VALUES))

        print('\nInserting ...\n')
        if all([len(VECS_s), len(VECS_o), len(VALUES)]):
            _MAX = len(VECS_words)
            pages = _MAX // _c
            if pages * _c < _MAX:
                pages += 1

            for p in tqdm(range(pages)):
                _start = p * _c
                _vecs_words = VECS_words[_start:_start + _c]
                _words = WORDS[_start:_start + _c]

                ids_words = milvusDB.insert(
                    np.mean(np.array(_vecs_words), axis=1),
                    dim=1024,
                    collection_name='words',
                    index_file_size=index_file_size,
                    metric='IP',
                    index='HNSW',
                )
                print(f'\n{len(ids_words)} words insert to milvus.\n')

                msg = zodb_insert_words(ids_words, _words)
                zodb_commit()
                print(f'\n{msg}\n')

            _MAX = len(VALUES)
            pages = _MAX // _c
            if pages * _c < _MAX:
                pages += 1

            _maxlen = max(max(max(len(_s), len(_o)) for _s, _o in zip(VECS_s, VECS_o)), 16)

            for p in tqdm(range(pages)):
                _start = p * _c

                _vecs_s = VECS_s[_start:_start + _c]
                _vecs_o = VECS_o[_start:_start + _c]
                _values = VALUES[_start:_start + _c]

                print('\nMilvus inserting ...\n')
                ids_dot = None
                if len(_vecs_s) > 0:
                    _vs_word_s = [VECS_words[WORDS.index(v.get('subject'))] for v in values]
                    _vs_word_o = [VECS_words[WORDS.index(v.get('object'))] for v in values]
                    relations = make_features(_maxlen, _vecs_s, _vecs_o, _vs_word_s, _vs_word_o)
                    print('Relation shape:', relations.shape)

                    ids_dot = milvusDB.insert(
                        relations,
                        dim=dim,
                        collection_name=db_code,
                        partition_tag=partition_tag,
                        index_file_size=index_file_size,
                        metric='IP',
                        index='HNSW',
                    )

                print(f'\n{len(_vecs_s)} records insert to milvus.\n')

                print('\nZODB inserting ...\n')
                msg = zodb_insert_rels(ids_dot, _values)
                zodb_commit()

                print(f'\n{msg}\n')

    return 'Done.'


if __name__ == '__main__':
    rst = ke_insert(new=args.new == 1)
    print(rst)
    db.close()
