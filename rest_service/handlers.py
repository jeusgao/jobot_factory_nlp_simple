#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
import atexit
import numpy as np
from fastapi import FastAPI
from itertools import product

from predictor import main
from utils import make_features
from storages import DIC_ZODB, milvusDB

app = FastAPI()
# app = FastAPI(on_shutdown=['on_shut'])

collection_dependency = 'dependency'
partition_tags_cosine = 'cosine'
partition_tags_dot = 'dot'

collection_words = 'words'
partition_tags_subjects = 'subjects'
partition_tags_objects = 'objects'
zodb_code = 'dependency'

db = DIC_ZODB.get(zodb_code)
db.open()


@atexit.register
def atexit_fun():
    db.close()
    print('ZODB closed.')


def normalization(x):
    return (x - np.min(x)) * 1 / (np.max(x) - np.min(x)) + 1e-9


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


def _get_qqp(text, _text_candidate):
    _r_qqp = main('qqp', text, _text_candidate)
    _is_match, _score = _r_qqp.get('result'), _r_qqp.get('score')

    return _score


@app.get("/{api_name}")
async def pred(api_name: str, input1: str, input2: str=None):
    rst = main(api_name, input1, input2)
    return rst


def _search_dot(_attns, _vecs_s, _vecs_o, values, top_k=5):
    rst = []
    simis_attn = milvusDB.search(
        _attns,
        collection_name=collection_dependency,
        partition_tags=partition_tags_dot,
        top_k=top_k,
    )
    simis_attn = [[(s.id, s.distance) for s in simi] for simi in simis_attn]

    simis_s = milvusDB.search(
        _vecs_s,
        collection_name=collection_words,
        partition_tags=partition_tags_subjects,
        top_k=1,
    )
    simis_o = milvusDB.search(
        _vecs_o,
        collection_name=collection_words,
        partition_tags=partition_tags_objects,
        top_k=1,
    )

    for simi_attn, simi_s, simi_o, value in zip(simis_attn, simis_s, simis_o, values):
        _s = db.root.dic_words[simi_s[0].id]
        _o = db.root.dic_words[simi_o[0].id]
        print(value.get('subject'), _s)
        print(value.get('object'), _o)
        print()
        _milvus_candidates = []
        for _id, _distance in simi_attn:
            _milvus_search_rst = None
            _key = f'{_s}{_id}{_o}'
            if _key in db.root.dic_pairs:
                _milvus_search_rst = db.root.dic_pairs[f'{_s}{_id}{_o}']
            if _milvus_search_rst:
                _milvus_search_rst['distance'] = _distance
                _text_candidate = _milvus_search_rst.get('text')
                _milvus_candidates.append(_milvus_search_rst)

        if len(_milvus_candidates) > 1:
            _milvus_candidates = list(sorted(_milvus_candidates, key=lambda x: x.get('distance'), reverse=True))

        value['candidates'] = _milvus_candidates
        del value['text']
        rst.append(value)
    return rst


@app.post("/ke_search")
async def ke_search(
    input1: str,
    input2: str=None,
    model: str='test',
    top_k: int=5,
):
    rst = main(model, input1, input2, from_api=False)
    text, words, rels = rst.get('text'), rst.get('words'), rst.get('rels')
    vecs_s, vecs_o, values = _get_rels(text, rels)

    _maxlen = max(max(len(_s), len(_o)) for _s, _o in zip(vecs_s, vecs_o))
    _attns, _vs_s, _vs_o = make_features(_maxlen, vecs_s, vecs_o)

    rst = _search_dot(_attns, _vs_s, _vs_o, values, top_k=top_k)

    return {'result': {'text': text, 'words': words, 'pairs': rst}}
