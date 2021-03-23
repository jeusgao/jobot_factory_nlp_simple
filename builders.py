#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 17:28:59
# @Author  : Joe Gao (jeusgao@163.com)


from backend import keras
from utils import get_object, DIC_DataLoaders
from modules import (
    DIC_Layers,
    DIC_Losses,
    DIC_Metrics,
    DIC_Models,
    DIC_Bases,
    DIC_Optimizers,
    DIC_Tokenizers,
    NonMaskingLayer,
    DIC_Funcs,
    DIC_Inits,
)


def model_builder(
    is_eval=False,
    is_predict=False,
    maxlen=128,
    ML=128,
    tokenizer_code=None,
    tokenizer_params={'fn_vocab': 'hub/bases/rbtl3/vocab.txt'},
    obj_common=None,
    dic_bases=None,
    dic_embeds=None,
    list_inputs=None,
    dic_layers=None,
    dic_outputs=None,
    obj_optimizer=None,
):
    dic_tokenizer = DIC_Tokenizers.get(tokenizer_code)
    token_dict, tokenizer = get_object(
        func=dic_tokenizer.get('func'),
        params=tokenizer_params,
    )

    _model_bases = {
        k: get_object(
            func=DIC_Bases.get(v.get('base_code')).get('func'),
            params={**v.get('base_params'), **{'seq_len': ML}}
        ) for k, v in dic_bases.items()
    }

    _model_embeds = {
        k: DIC_Models.get(v.get('model_type')).get('func')(_model_bases.get(v.get('base'))) for k, v in dic_embeds.items()
    }

    def _get_IOS(_layers, list_IOS, key_type, key, tag):
        _IOS = []
        for _d in list_IOS:
            _IOS_type = _d.get(key_type)
            _IOS_code = _d.get(key)

            _IO = None
            if _IOS_type == 'Pretrained':
                if tag == 'O':
                    _IO = _model_bases.get(_IOS_code).output
                else:
                    _IO = _model_bases.get(_IOS_code).inputs
            if _IOS_type == 'Embeded':
                if tag == 'O':
                    _IO = _model_embeds.get(_IOS_code).output
                else:
                    _IO = _model_embeds.get(_IOS_code).inputs
            if _IOS_type == 'Layer':
                _IO = _layers.get(_IOS_code)

            if is_predict or (not is_predict and not _d.get('for_pred_only')):
                _IOS.append(_IO)

        return _IOS

    _model_layers = {}
    for k, v in dic_layers.items():
        _layer_type = DIC_Layers.get(v.get('layer')).get('func')

        _params = v.get('params')

        if _params and 'kernel_initializer' in _params:
            _params['kernel_initializer'] = DIC_Inits.get(_params.get('kernel_initializer')).get('func')()

        if _params and isinstance(_params, str):
            _params = DIC_Funcs.get(_params)

        _inputs = _get_IOS(_model_layers, v.get('layer_inputs'), 'inputs_type', 'inputs', 'O')
        if len(_inputs) == 1:
            _inputs = _inputs[0]

        _layer = get_object(func=_layer_type if _params or 'lambda' in v.get(
            'layer') or 'crf' in v.get('layer') else _layer_type(), params=_params)(_inputs)
        _model_layers[k] = _layer

    _model_inputs = _get_IOS(_model_layers, list_inputs, 'inputs_type', 'inputs', 'I')
    _model_outputs = _get_IOS(_model_layers, dic_outputs.values(), 'output_type', 'output', 'O')
    _model_losses = [DIC_Losses.get(v.get('loss')).get('func') for v in dic_outputs.values() if v.get('loss')]
    _model_metrics = [[DIC_Metrics.get(_metric).get('func') for _metric in v.get('metrics')]
                      for v in dic_outputs.values() if len(v.get('metrics')) > 0]

    model = keras.Model(_model_inputs, _model_outputs)
    optimizer = get_object(func=DIC_Optimizers.get(obj_optimizer.func).get('func'), params=obj_optimizer.params)
    _model_loss_weights = [1., 1.] if is_eval else obj_optimizer.loss_weights

    model.compile(
        optimizer='adam' if is_eval else optimizer,
        loss=_model_losses,
        loss_weights=_model_loss_weights,
        metrics=_model_metrics,
    )

    # model.summary()
    return tokenizer, token_dict, model


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
    # data_loader_params = data_loader_params.get('params')

    _x, _y = data_loader(**{'fns': fns})
    _steps = len(_x) // batch_size

    return _x, _y, _steps
