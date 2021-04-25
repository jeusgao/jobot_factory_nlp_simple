#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 11:33:20
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import numpy as np
import scipy.sparse as sp
from collections import namedtuple


def get_object(func=None, params=None):
    if params:
        if isinstance(params, dict):
            return func(**params)
        else:
            return func(params)
    else:
        return func


def get_namedtuple(tag, dic):
    CFG_MODEL = namedtuple(tag, dic.keys())
    return CFG_MODEL(**dic)


def task_init(task_path, is_train=True):
    if is_train:
        pid = os.getpid()
        with open(f'{task_path}/training.pid', 'w') as f:
            f.write(pid.__str__())

    fn_model = f'{task_path}/model.h5'

    dic_task_params = {}

    def _set_task_params(fn, param, mod='json'):
        if os.path.exists(fn):
            with open(fn) as f:
                if mod == 'json':
                    dic_task_params[param] = json.load(f)
                else:
                    dic_task_params[param] = get_namedtuple(param, json.load(f))
        else:
            dic_task_params[param] = None

    _set_task_params(f'{task_path}/model_bases_params.json', 'model_bases_params')
    _set_task_params(f'{task_path}/model_common_params.json', 'model_common_params', mod='namedtuple')
    _set_task_params(f'{task_path}/model_embeded_params.json', 'model_embeded_params')
    _set_task_params(f'{task_path}/model_inputs_params.json', 'model_inputs_params')
    _set_task_params(f'{task_path}/model_layer_params.json', 'model_layer_params')
    _set_task_params(f'{task_path}/model_outputs_params.json', 'model_outputs_params')
    _set_task_params(f'{task_path}/model_optimizer_params.json', 'model_optimizer_params', mod='namedtuple')
    _set_task_params(f'{task_path}/params_data.json', 'params_data', mod='namedtuple')
    _set_task_params(f'{task_path}/params_train.json', 'params_train', mod='namedtuple')
    _set_task_params(f'{task_path}/params_pred.json', 'params_pred', mod='namedtuple')

    return fn_model, dic_task_params


def get_params(fn):
    with open(fn, 'r') as f:
        params = json.load(f)
    return params


def get_lines(fn):
    lines = []
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.read().splitlines()
    return lines


def get_dic_from_json(fn):
    dic = None
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            dic = json.load(f)
    return dic


def get_key_from_json(fn, key):
    _v = False
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            _v = json.load(f)
        _v = _v.get(key)
    return _v


def kill_process(fn):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            pid = f.read()
        cmd = f'kill -9 {pid}'
        os.system(cmd)


def dump_json(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


def preprocess_features(x):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    x = r_mat_inv.dot(x)
    # return features.todense()
    return x


def leaky_relu(x, alpha=0.2):
    return np.maximum(0.2 * x, x)


def softmax_1d(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def attention(a=None, b=None):
    features = preprocess_features(a)
    self_attn = np.multiply(features, a)
    neigh_attn = np.multiply(features, b)
    dense = self_attn + neigh_attn
    dense_leaky = leaky_relu(dense)
    # soft_max = softmax(dense_leaky)

    return dense_leaky


def make_features(_maxlen, _vs_s, _vs_o, _vs_word_s, _vs_word_o, mod='cosine'):
    _vs_s = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vs_s]
    _vs_o = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vs_o]

    _vs_word_s = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vs_word_s]
    _vs_word_o = [np.pad(_v, ((0, _maxlen - len(_v)), (0, 0)), 'mean') for _v in _vs_word_o]

    if mod == 'cosine':
        relationships = np.mean(
            np.cos(
                np.cos(np.array(_vs_s), np.array(_vs_o)),
                np.cos(np.array(_vs_word_s), np.array(_vs_word_o))
            ),
            axis=1
        )
    else:
        _attn_vs_s = np.array([attention(a=_s, b=_o) for _s, _o in zip(_vs_s, _vs_o)])
        _attn_vs_o = np.array([attention(a=_o, b=_s) for _s, _o in zip(_vs_s, _vs_o)])

        relationships = np.mean(np.multiply(_attn_vs_s, _attn_vs_o), axis=1)

    return relationships


def get_rels(text, rels):
    vecs_s, vecs_o, values = [], [], []
    for r in rels:
        subj = r.get('from_word')
        pos_subj = r.get('from_pos')
        obj = r.get('to_word')
        pos_obj = r.get('to_pos')
        if len(subj) > 0 and len(obj) > 0:
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
