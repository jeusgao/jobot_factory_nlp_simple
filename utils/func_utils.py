#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 11:33:20
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json


def get_object(func=None, params=None):
    return func(**params) if params else func()


def trainer_init(task_path):
    pid = os.getpid()
    with open(f'{task_path}/training.pid', 'w') as f:
        f.write(pid.__str__())

    fn_model = f'{task_path}/model.h5'

    with open(f'{task_path}/params_model.json') as f:
        params_model = json.load(f)

    with open(f'{task_path}/params_data.json') as f:
        params_data = json.load(f)

    with open(f'{task_path}/params_train.json') as f:
        params_train = json.load(f)

    return fn_model, params_model, params_data, params_train


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
