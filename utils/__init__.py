#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

from .callbacks import TrainingCallbacks, EvaluatingCallbacks
from .data_utils import (
    sentences_loader_train,
    sequences_loader_train,
    data_generator_train,
    data_generator_eval,
    data_generator_pred,
)
from .func_utils import (
    get_object,
    trainer_init,
    get_params,
    get_lines,
    get_dic_from_json,
    get_key_from_json,
    kill_process,
    dump_json,
)

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
    }
}

DIC_Generators = {
    'data_generator_train': {
        'func': data_generator_train,
        'params': {
            'data': None,
            'Y': None,
            'tokenizer': None,
            'dim': 2,
            'maxlen': 512,
            'labeler': None,
            'activation': 'sigmoid',
        },
    },
    'data_generator_pred': {
        'func': data_generator_pred,
    }
}
