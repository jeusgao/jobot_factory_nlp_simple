#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 11:33:20
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
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

    with open(f'{task_path}/model_bases_params.json') as f:
        dic_task_params['model_bases_params'] = json.load(f)

    with open(f'{task_path}/model_common_params.json') as f:
        dic_task_params['model_common_params'] = get_namedtuple('model_common_params', json.load(f))

    if os.path.exists(f'{task_path}/model_embeded_params.json'):
        with open(f'{task_path}/model_embeded_params.json') as f:
            dic_task_params['model_embeded_params'] = json.load(f)
    else:
        dic_task_params['model_embeded_params'] = None

    with open(f'{task_path}/model_inputs_params.json') as f:
        dic_task_params['model_inputs_params'] = json.load(f)

    with open(f'{task_path}/model_layer_params.json') as f:
        dic_task_params['model_layer_params'] = json.load(f)

    with open(f'{task_path}/model_outputs_params.json') as f:
        dic_task_params['model_outputs_params'] = json.load(f)

    with open(f'{task_path}/model_optimizer_params.json') as f:
        dic_task_params['model_optimizer_params'] = get_namedtuple('model_optimizer_params', json.load(f))

    with open(f'{task_path}/params_data.json') as f:
        dic_task_params['params_data'] = get_namedtuple('params_data', json.load(f))

    with open(f'{task_path}/params_train.json') as f:
        dic_task_params['params_train'] = get_namedtuple('params_train', json.load(f))

    with open(f'{task_path}/params_pred.json') as f:
        dic_task_params['params_pred'] = get_namedtuple('params_pred', json.load(f))

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
    with open(fn, 'r') as f:
        pid = f.read()
    cmd = f'kill -9 {pid}'
    os.system(cmd)


def dump_json(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)
