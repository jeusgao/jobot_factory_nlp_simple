#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-12 14:08:32
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
from .components import get_default


def output_conf(c2, _dic, _dic_outputs_types, _key, _model_output_params, DIC_Losses, DIC_Metrics):
    _dic_value = {}
    _model_output_value_params = _model_output_params.get(_key, {}) if _model_output_params else {}

    _options, _default = [False, True], _model_output_value_params.get('for_pred_only', False)
    _dic_value['for_pred_only'] = c2.selectbox('is output for predictor only', _options, _options.index(_default))

    _options, _default = get_default(_model_output_value_params, _dic_outputs_types, 'output_type')
    _outputs_type = c2.selectbox('model output type', _options, _options.index(_default))
    _dic_value['output_type'] = _outputs_type

    _dic_outputs_type = _dic_outputs_types.get(_outputs_type)
    _options = list(_dic_outputs_type.keys()) if _dic_outputs_type else ['']
    _options, _default = get_default(_model_output_value_params, _dic_outputs_type, 'output')
    _default = _options.index(_default) if _default in _options else 0
    _dic_value['output'] = c2.selectbox('model output', _options, _default)

    _options, _default = get_default(_model_output_value_params, DIC_Losses, 'loss')
    _options = _options + [None]
    _dic_value['loss'] = c2.selectbox('model output loss', _options, _options.index(_default))

    _options = list(DIC_Metrics.keys())
    _default = _model_output_value_params.get('metrics', [])
    _dic_value['metrics'] = c2.multiselect(
        'Select metrics', _options,
        default=_default if set(_options).intersection(set(_default)) else _options[0],
        key='metrics',
    )

    _dic[_key] = _dic_value
    _dic = {k: v for k, v in sorted(_dic.items(), key=lambda x: x[0])}

    return _dic


def add_output(c2, _dic_outputs_types, _model_output_params, DIC_Losses, DIC_Metrics):
    _key = c2.text_input('Input a new name:', '').strip()
    if not len(_key):
        return False, 'Input a name please.'
    if _key in _model_output_params:
        return False, f'Duplcated name - [{_key}] .'

    _dic = {k: v for k, v in _model_output_params.items()} if _model_output_params else {}
    _dic = output_conf(c2, _dic, _dic_outputs_types, _key, _model_output_params, DIC_Losses, DIC_Metrics)
    return True, _dic
