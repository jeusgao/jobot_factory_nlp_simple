#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-16 14:47:23
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json

from utils import (
    DIC_DataLoaders,
    DIC_Generators_for_train,
    DIC_Generators_for_pred,
    DIC_Resolvers,
)
from modules import(
    DIC_Losses,
    DIC_Metrics,
    DIC_Layers,
    DIC_Bases,
    DIC_Optimizers,
    DIC_Tokenizers,
)


def _get_dic(_dic):
    _d = {}
    for k, v in _dic.items():
        _d[k] = {}
        _d[k]['func'] = ''
        if v.get('params'):
            _d[k]['params'] = v.get('params')
    return _d


_dics = {
    'DIC_Losses': _get_dic(DIC_Losses),
    'DIC_Metrics': _get_dic(DIC_Metrics),
    'DIC_Layers': _get_dic(DIC_Layers),
    'DIC_Bases': _get_dic(DIC_Bases),
    'DIC_Optimizers': _get_dic(DIC_Optimizers),
    'DIC_Tokenizers': _get_dic(DIC_Tokenizers),
    'DIC_DataLoaders': _get_dic(DIC_DataLoaders),
    'DIC_Generators_for_train': _get_dic(DIC_Generators_for_train),
    'DIC_Generators_for_pred': _get_dic(DIC_Generators_for_pred),
    'DIC_Resolvers': _get_dic(DIC_Resolvers),
}


def env_init():
    with open('params_templates.json', 'w') as f:
        json.dump(_dics, f, ensure_ascii=False, indent=2)

    if not os.path.exists('hub/bases'):
        os.makedirs('hub/base')

    if not os.path.exists('hub/models'):
        os.makedirs('hub/models')

    if not os.path.exists('data'):
        os.mkdir('data')


if __name__ == '__main__':
    env_init()
