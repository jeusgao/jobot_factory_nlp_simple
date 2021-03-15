#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-11 16:18:33
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
from utils import get_dic_from_json

_dics = get_dic_from_json('params_templates.json')

DIC_DataLoaders = _dics.get('DIC_DataLoaders')
DIC_Generators_for_train = _dics.get('DIC_Generators_for_train')
DIC_Generators_for_pred = _dics.get('DIC_Generators_for_pred')
DIC_Layers = _dics.get('DIC_Layers')
DIC_Losses = _dics.get('DIC_Losses')
DIC_Metrics = _dics.get('DIC_Metrics')
DIC_Bases = _dics.get('DIC_Bases')
DIC_Optimizers = _dics.get('DIC_Optimizers')
DIC_Tokenizers = _dics.get('DIC_Tokenizers')
DIC_Models = _dics.get('DIC_Models')

dic_embed_models = {
    'model_type': 'DIC_Models',
    'inputs_src': 'Base Model',
    'inputs_key': 'str'
}

dic_inputs = [
    {
        'is_multi': True,
        'key': 'func',
        'value': {
            'type': 'select',
            'input': [
                ('Base Model', 'Inputs'),
                ('Embed Model', 'Inputs'),
                ('Model Layer', None),
            ],
        },
    },
]

dic_func = {'func': '', 'params': '{"": ""}'}

dic_layers = [
    {
        'is_multi': True,
        'key': 'func',
        'value': {
            'type': 'select',
            'input': DIC_Layers,
        },
    },
    {
        'key': 'params',
        'is_multi': False,
        'value': {
            'type': 'text',
            'input': '{"": ""}',
        },
    },
    {
        'key': 'layer_inputs',
        'is_multi': True,
        'value': {
            'type': 'select',
            'input': dic_inputs,
        },
    },
]

dic_model_outputs = {
    'func': '',
    'loss': {'func': False, 'dic': dic_func},
    'metrics': {'func': True, 'dic': dic_func},
}
