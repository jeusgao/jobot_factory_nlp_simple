#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
import numpy as np
from fastapi import FastAPI

from predictor import main
from storages import DIC_ZODB, milvusDB

app = FastAPI()

collection_cosine = 'dependency'
partition_tags_cosine = '202103'

collection_words = 'words'
partition_tags_subjects = 'subjects'
partition_tags_objects = 'objects'
zodb_code = 'dependency'

db = DIC_ZODB.get(zodb_code)
db.open()

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


# @app.post("/ke_insert")
# async def ke_insert(
#     input1: str,
#     input2: str=None,
#     dim: int=2048,
#     model: str='test',
#     db_code: str='dependency',
#     partition_tag: str='202103',
#     index_file_size: int=1024,
# ):
#     rst = main(model, input1, input2)

#     text, rels = rst.get('text'), rst.get('rels')
#     vecs, values = _get_rels(text, rels)

#     ids = milvusDB.insert(
#         vecs,
#         dim=dim,
#         collection_name=db_code,
#         partition_tag=partition_tag,
#         index_file_size=index_file_size,
#     )
#     milvusDB.commit()

#     db = DIC_ZODB.get(db_code)
#     msg = db.insert(ids, values)

#     return {'result': msg}

def search_cosine(_cosines, values, top_k=5):
    rst = []

    simis_cosine = milvusDB.search(
        _cosines,
        collection_name=collection_cosine,
        partition_tags=partition_tags_cosine,
        top_k=top_k
    )
    simis_cosine = [[(s.id, s.distance) for s in simi] for simi in simis_cosine]

    for simi, value in zip(simis_cosine, values):
        _milvus_candidates = []
        for _id, _distance in simi:
            _milvus_search_rst = db.root.dic_cosines[_id]
            _milvus_search_rst['distance'] = _distance
            _text_candidate = _milvus_search_rst.get('text')

            # qqp_score = _get_qqp(text, _text_candidate)
            # _milvus_search_rst['text_match_score'] = qqp_score

            _milvus_candidates.append(_milvus_search_rst)

        _milvus_candidates = list(sorted(_milvus_candidates, key=lambda x: x.get('distance')))

        value['candidates'] = _milvus_candidates
        del value['text']
        rst.append(value)
    return rst


def search_pairs(_vs_s, _vs_o, _cosines, values, top_k=5):
    rst = []
    simis_subjects = milvusDB.search(np.mean(np.array(_vs_s), axis=1), collection_name=collection_words,
                                     partition_tags=partition_tags_subjects, top_k=top_k)
    simis_subjects = [[(s.id, s.distance) for s in simi] for simi in simis_subjects]
    simis_subjects = list(sorted(simis_subjects, key=lambda x: x[1]))

    simis_objects = milvusDB.search(np.mean(np.array(_vs_o), axis=1), collection_name=collection_words,
                                    partition_tags=partition_tags_objects, top_k=top_k)
    simis_objects = [[(s.id, s.distance) for s in simi] for simi in simis_objects]
    simis_objects = list(sorted(simis_objects, key=lambda x: x[1]))

    all_ids_cosine = []
    for simis_subject, simis_object, _cosine, value in zip(simis_subjects, simis_objects, _cosines, values):
        _cosine_candidates = []
        for simi_s, simi_o in zip(simis_subject, simis_object):
            id_s, dist_s = simi_s
            id_o, dist_o = simi_o
            _key = (id_s, id_o)
            if _key in db.root.dic_pairs.keys():
                id_cosine = db.root.dic_pairs[_key]
                _entity = milvusDB.db.get_entity_by_id(collection_cosine, [id_cosine])[1][0]
                cosine_simi = np.mean(np.cos(_cosine, np.array(_entity)))
                _value = db.root.dic_cosines[id_cosine]
                _value['distance'] = cosine_simi
                _cosine_candidates.append(_value)
        if len(_cosine_candidates):
            value['candidates'] = _cosine_candidates
            del value['text']
            rst.append(value)
    return rst


@app.post("/ke_search")
async def ke_search(
    input1: str,
    input2: str=None,
    search_mod: str='cosine',
    model: str='test',
    top_k: int=5,
):
    rst = main(model, input1, input2, from_api=False)
    text, words, rels = rst.get('text'), rst.get('words'), rst.get('rels')
    vecs_s, vecs_o, values = _get_rels(text, rels)

    _maxlen = max(max(len(_s), len(_o)) for _s, _o in zip(vecs_s, vecs_o))

    _vs_s = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in vecs_s]
    _vs_o = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in vecs_o]

    _cosines = np.mean(np.cos(np.array(_vs_s), np.array(_vs_o)), axis=1)

    if search_mod == 'cosine':
        rst = search_cosine(_cosines, values, top_k=top_k)
    else:
        rst = search_pairs(_vs_s, _vs_o, _cosines, values, top_k=top_k)
        # if not len(rst):
        #     rst = search_cosine(_cosines, values, top_k=top_k)

    return {'result': {'text': text, 'words': words, 'pairs': rst}}
