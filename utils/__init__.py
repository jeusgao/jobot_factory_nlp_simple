#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

from .data_utils import (
    sentences_loader_train,
    sequences_loader_train,
    rel_data_loader,
)
from .func_utils import (
    get_object,
    task_init,
    get_params,
    get_lines,
    get_dic_from_json,
    get_key_from_json,
    kill_process,
    dump_json,
)
from .resolvers import resolve, resolve_spo

DIC_Resolvers = {
    'resolver': {
        'func': resolve
    },
    'spo_resolver': {
        'func': resolve_spo
    },
}

DIC_DataLoaders = {
    'sentences_loader_train': {
        'func': sentences_loader_train,
        'params': {
            'dir_data': None,
            'data_cls': None,
            't1': 'q1',
            't2': None,
            'label': 'label',
        }
    },
    'sequences_loader_train': {
        'func': sequences_loader_train,
        'params': {
            'dir_data': None,
            'data_cls': None,
        },
    },
    'relationship_loader_train': {
        'func': rel_data_loader,
        'params': {
            'dir_data': None,
            'data_cls': None,
        },
    },
}
