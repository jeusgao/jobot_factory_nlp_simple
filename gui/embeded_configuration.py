#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-12 15:30:34
# @Author  : Joe Gao (jeusgao@163.com)

import os
from .components import get_default


def embed_conf(c2=None, _dic=None, _embeded_model_params=None, _base_model_params=None, DIC_Models=None, _key=None):
    _embeded_model_value_params = _embeded_model_params.get(_key) if _embeded_model_params else None
    _dic_value = {}

    _options, _default = get_default(_embeded_model_value_params, DIC_Models, 'model_type')
    _dic_value['model_type'] = c2.selectbox('model type', _options, _options.index(_default))

    _options, _default = get_default(_embeded_model_value_params, _base_model_params, 'base')
    _dic_value['base'] = c2.selectbox('base model', _options, _options.index(_default))

    _dic[_key] = _dic_value
    _dic = {k: v for k, v in sorted(_dic.items(), key=lambda x: x[0])}

    return _dic


def pretrained_conf(c2=None, _key=None, _dic=None, _pretrained_model_params=None, DIC_Bases=None):
    _pretrained_model_value_params = _pretrained_model_params.get(_key) if _pretrained_model_params else None
    _dic_value = {}

    _options, _default = get_default(_pretrained_model_value_params, DIC_Bases, 'base_code')
    _dic_value['base_code'] = c2.selectbox('base type', _options, _options.index(_default))

    _options = tuple(os.walk('hub/bases'))[0][1]
    _default = _options[0]
    if _pretrained_model_value_params:
        _default = _pretrained_model_value_params.get('base_params').get('fn_config').split('/')[-2]
    _base_path = c2.selectbox('base model', _options, _options.index(_default))

    _dic_value['base_params'] = {
        'fn_config': f'hub/bases/{_base_path}/bert_config.json',
        'fn_base_model': f'hub/bases/{_base_path}/bert_model.ckpt',
        'training': c2.radio('is base training', [True, False], index=1),
        'trainable': c2.radio('is base trainable', [True, False], index=0),
    }

    _dic[_key] = _dic_value
    _dic = {k: v for k, v in sorted(_dic.items(), key=lambda x: x[0])}

    return _dic
