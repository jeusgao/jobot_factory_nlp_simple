#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
import numpy as np
from fastapi import FastAPI

from predictor import main
from storages import DIC_ZODB, milvusDB

app = FastAPI()


'''ZODB API
# from pydantic import BaseModel

# class ZodbItem(BaseModel):
#     key: int
#     subj: str
#     obj: str
#     score: float
#     text: str

# @app.post("/zodb_insert")
# async def zodb_insert(item: ZodbItem, db_code: str='dependency'):
#     db = DIC_ZODB.get(db_code)
#     key = item.key
#     value = {'text': item.text, 'subject': item.subj, 'object': item.obj, 'score': item.score}
#     msg = db.insert(key, value)
#     return {'result': msg}

# @app.post("/zodb_search")
# async def zodb_search(key: int, db_code: str='dependency'):
#     db = DIC_ZODB.get(db_code)
#     return {'result': db.search(key)}
'''


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


@app.get("/{api_name}")
async def pred(api_name: str, input1: str, input2: str=None):
    rst = main(api_name, input1, input2)
    return rst


@app.post("/ke_insert")
async def ke_insert(
    input1: str,
    input2: str=None,
    dim: int=2048,
    model: str='test',
    db_code: str='dependency',
    partition_tag: str='202103',
    index_file_size: int=1024,
):
    rst = main(model, input1, input2)

    text, rels = rst.get('text'), rst.get('rels')
    vecs, values = _get_rels(text, rels)

    ids = milvusDB.insert(
        vecs,
        dim=dim,
        collection_name=db_code,
        partition_tag=partition_tag,
        index_file_size=index_file_size,
    )
    milvusDB.commit()

    db = DIC_ZODB.get(db_code)
    msg = db.insert(ids, values)

    return {'result': msg}


@app.post("/ke_search")
async def ke_search(
    input1: str,
    input2: str=None,
    model: str='test',
    db_code: str='dependency',
    partition_tags: str=None,
    top_k: int=5,
):
    db = DIC_ZODB.get(db_code)
    db.open()

    rst = main(model, input1, input2)
    text, words, rels = rst.get('text'), rst.get('words'), rst.get('rels')
    vecs, values = _get_rels(text, rels)

    simis = milvusDB.search(vecs, collection_name=db_code, partition_tags=partition_tags, top_k=top_k)
    simis = [[(s.id, s.distance) for s in simi] for simi in simis]

    rst = []

    for simi, value in zip(simis, values):
        _milvus_candidates = []
        for _id, _distance in simi:
            _milvus_search_rst = db.search(_id)
            _milvus_search_rst['distance'] = _distance
            _text_candidate = _milvus_search_rst.get('text')

            _r_qqp = main('qqp', text, _text_candidate)

            _is_match, _score = _r_qqp.get('result'), _r_qqp.get('score')
            _milvus_search_rst['text_match_score'] = _score

            _milvus_candidates.append(_milvus_search_rst)

        value['candidates'] = _milvus_candidates
        del value['text']
        rst.append(value)

    db.close_conn()

    return {'result': {'text': text, 'words': words, 'pairs': rst}}
