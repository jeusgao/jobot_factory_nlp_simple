#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-12 14:08:32
# @Author  : Joe Gao (jeusgao@163.com)

import os
from .components import get_default


def inputs_conf(c2, st, _default, _inputs, _dic_inputs_types):
    _num_inputs = c2.number_input('numer of inputs:', min_value=0, max_value=20, value=_default, step=1)
    ls = []
    for i in range(_num_inputs):
        with c2.beta_expander(f"Input No. {i}"):
            _options = ['Pretrained', 'Embeded', 'Layer']
            _default = _inputs[i].get('inputs_type', 'Pretrained') if i < len(
                _inputs) and _inputs[i] else _options[0]
            _inputs_type = st.selectbox(
                'select inputs source',
                _options,
                _options.index(_default),
                key=f'inputs source_{i}'
            )

            _dic_inputs_type = _dic_inputs_types.get(_inputs_type)
            _options = list(_dic_inputs_type.keys()) if _dic_inputs_type else ['']

            _cur = _inputs[i].get('inputs') if i < len(_inputs) else None
            _default = _options.index(_cur) if i < len(_inputs) and _cur and _cur in _options else 0
            _layer_inputs = st.selectbox(
                'select inputs',
                _options,
                _default,
                key=f'inputs_{i}',
            )
            if len(_layer_inputs.strip()):
                ls.append({'inputs_type': _inputs_type, 'inputs': _layer_inputs})
    return ls


def layer_conf(c2, st, _dic, _dic_inputs_types, _key, _model_layer_params, DIC_Layers, _options=None):
    _dic_value = {}
    _model_layer_value_params = _model_layer_params.get(_key) if _model_layer_params else None

    _options, _default = get_default(_model_layer_value_params, DIC_Layers, 'layer')
    _dic_value['layer'] = c2.selectbox('model layer', _options, _options.index(_default))

    _default = _model_layer_value_params.get('params', '') if _model_layer_value_params else None

    _layer_params = c2.text_input('layer params', _default)
    if not len(_layer_params.strip()) or _layer_params == 'None':
        _layer_params = None
    if _layer_params:
        if '{' in _layer_params or '[' in _layer_params:
            try:
                _dic_value['params'] = eval(_layer_params)
            except Exception as err:
                c2.error(f'{err}, Check your input please...')
        else:
            _dic_value['params'] = _layer_params

    _inputs = _model_layer_value_params.get('layer_inputs', []) if _model_layer_value_params else []
    _default = len(_inputs) if _inputs else 0

    ls = inputs_conf(c2, st, _default, _inputs, _dic_inputs_types)
    _dic_value['layer_inputs'] = ls
    _dic[_key] = _dic_value

    _dic = {k: v for k, v in sorted(_dic.items(), key=lambda x: x[0])}

    return _dic


def add_layer(c2, st, _dic_inputs_types, _model_layer_params, DIC_Layers):
    _key = c2.text_input('Input a new name:', '').strip()
    if not len(_key):
        return False, 'Input a name please.'
    if _model_layer_params and _key in _model_layer_params:
        return False, f'Duplcated name - [{_key}] .'

    _dic = {k: v for k, v in _model_layer_params.items()} if _model_layer_params else {}
    _dic = layer_conf(c2, st, _dic, _dic_inputs_types, _key, _model_layer_params, DIC_Layers)
    return True, _dic
