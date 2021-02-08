#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 17:28:59
# @Author  : Joe Gao (jeusgao@163.com)

from collections import namedtuple

from backend import keras
from utils import get_object, DIC_DataLoaders
from modules import (
    DIC_Layers,
    DIC_Losses,
    DIC_Metrics,
    DIC_Bases,
    DIC_Optimizers,
    DIC_Tokenizers,
)


def model_builder(
    is_eval=False,
    TF_KERAS=0,
    maxlen=64,
    ML=64,
    tokenizer_code='tokenizer_zh',
    tokenizer_params=None,
    base_code='BERT',
    base_params=None,
    model_params=None,
):
    dic_tokenizer = DIC_Tokenizers.get(tokenizer_code)
    toker_dict, tokenizer = get_object(
        func=dic_tokenizer.get('func'),
        params=tokenizer_params,
    )

    dic_base = DIC_Bases.get(base_code)
    base_params['seq_len'] = ML
    base = get_object(
        func=dic_base.get('func'),
        params=base_params,
    )

    CFG_MODEL = namedtuple('CFG_MODEL', model_params.keys())
    cfg_model = CFG_MODEL(**model_params)

    inputs = base.inputs
    output = base.output

    for layer in cfg_model.layers:
        print(layer)
        params = layer.get('params', None)
        if params:
            output = get_object(
                func=DIC_Layers.get(layer.get('func')).get('func'),
                params=params)(output)
        else:
            output = DIC_Layers.get(layer.get('func')).get('func')(output)
    model = keras.Model(inputs, output)

    _loss = cfg_model.loss
    loss = DIC_Losses.get(_loss.get('func')).get('func')
    if 'params' in _loss.keys():
        loss = get_object(func=loss, params=_loss.get('params'))

    metrics = cfg_model.metrics
    metrics = [
        get_object(func=DIC_Metrics.get(m.get('func')).get('func'), params=m.get('params'))
        if 'params' in DIC_Metrics.get(m.get('func')).keys()
        else DIC_Metrics.get(m.get('func')).get('func')
        for m in metrics
    ]

    _optimizer = cfg_model.optimizer
    optimizer = DIC_Optimizers.get(_optimizer.get('func'))
    optimizer = get_object(func=optimizer.get('func'), params=_optimizer.get('params'))

    model.compile(
        optimizer='adam' if is_eval else optimizer,
        loss=loss,
        metrics=metrics,
    )

    # model.summary()
    return tokenizer, model


def train_data_builder(
    data_loader_params={
        'func': 'sentences_loader_train',
        'params': {
            'dir_data': 'data',
            'data_cls': 'qqp',
            't1': 'q1',
            't2': 'q2',
            'label': 'label',
        },
    },
    fns=['test_lcqmc.tsv'],
    batch_size=128,
):
    data_loader = DIC_DataLoaders.get(data_loader_params.get('func')).get('func')
    data_loader_params = data_loader_params.get('params')

    _x, _y = data_loader(**{**{'fns': fns}, **data_loader_params})
    _steps = len(_x) // batch_size

    return _x, _y, _steps
