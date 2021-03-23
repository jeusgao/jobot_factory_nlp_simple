#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-13 13:00:47
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import glob
import streamlit as st

from utils import get_dic_from_json, dump_json
from .layer_configuration import inputs_conf, layer_conf, add_layer
from .embeded_configuration import embed_conf, pretrained_conf
from .output_configuration import output_conf, add_output
from .components import (
    get_default,
    get_default_input,
    get_default_params,
    multi_options,
    crud,
)

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
DIC_Resolvers = _dics.get('DIC_Resolvers')

fn_tmp_layers = 'tmp_layers.txt'
fn_tmp_metrics = 'tmp_metrics.txt'


def model_params(task_path, is_training=False):
    # _model_params = get_dic_from_json(f'{task_path}/params_model.json')
    _dic = {}

    titles = st.title('Model params')

    c1, c2 = st.beta_columns([1, 2])
    _choice = c1.radio('Select a part to edit:', [
        'Common Params',
        'Pretrained Models',
        'Embeded Models',
        'Model Layers',
        'Model Inputs',
        'Model Outputs',
        'Model Optimizer',
    ])

    if _choice == 'Common Params':
        _common_params = get_dic_from_json(f'{task_path}/model_common_params.json')
        with c2.beta_expander("Backend and LR"):
            _default = _common_params.get('TF_KERAS') if _common_params else 0
            _dic['TF_KERAS'] = st.radio('Use TF-Keras?', [0, 1], _default)
            _default = _common_params.get('LR') if _common_params else None
            _dic['LR'] = float(
                st.number_input(
                    'Model learning rate:',
                    min_value=0.,
                    max_value=0.1,
                    value=0.00002,
                    step=0.000001,
                    format='%f'
                )
            )
        c2.subheader('')
        with c2.beta_expander("Inputs params"):
            maxlen = st.number_input(
                'input max lenth',
                min_value=8,
                max_value=512,
                value=_common_params.get('maxlen') if _common_params else 32,
                step=8,
            )
            _dic['maxlen'] = maxlen

            _default = True
            if _common_params and 'is_pair' in _common_params:
                _default = False

            is_pair = st.radio('is pair inputs', [True, False], [True, False].index(_default))
            _dic['ML'] = maxlen * 2 + 3 if is_pair else maxlen
            _dic['is_pair'] = is_pair

        c2.subheader('')
        if not is_training and c2.button('save'):
            dump_json(f'{task_path}/model_common_params.json', _dic)
            c2.success('Params saved.')

    if _choice == 'Pretrained Models':
        _pretrained_model_params = get_dic_from_json(f'{task_path}/model_bases_params.json')
        _options = None
        if _pretrained_model_params:
            _options = list(_pretrained_model_params.keys())
            _dic = {k: v for k, v in _pretrained_model_params.items()}
        _key = ''

        _status, _dic, _msg, is_delete = crud(
            c2,
            _pretrained_model_params,
            pretrained_conf,
            {'c2': c2, '_key': _key, '_dic': _dic, '_pretrained_model_params': _pretrained_model_params, 'DIC_Bases': DIC_Bases},
            'Pretrained',
        )
        if _status:
            c2.subheader('')
            if is_delete:
                dump_json(f'{task_path}/model_bases_params.json', _dic)
                _pretrained_model_params = _dic
                c2.success('Selected pretrained model removed from this model.')
            else:
                if not is_training and c2.button('save', key='pretrained_model_params'):
                    dump_json(f'{task_path}/model_bases_params.json', _dic)
                    _pretrained_model_params = _dic
                    c2.success(_msg)
        else:
            if _msg:
                c2.warning(_msg)

        st.write(_pretrained_model_params)

    if _choice == 'Embeded Models':
        _base_model_params = get_dic_from_json(f'{task_path}/model_bases_params.json')
        _embeded_model_params = None
        if _base_model_params:
            _embeded_model_params = get_dic_from_json(f'{task_path}/model_embeded_params.json')
            _options, _dic = None, {}
            if _embeded_model_params:
                _options = list(_embeded_model_params.keys())
                _dic = {k: v for k, v in _embeded_model_params.items()}
            _key = ''

            _status, _dic, _msg, is_delete = crud(
                c2,
                _embeded_model_params,
                embed_conf,
                {'c2': c2, '_dic': _dic, '_embeded_model_params': _embeded_model_params,
                    '_base_model_params': _base_model_params, 'DIC_Models': DIC_Models, '_key': _key},
                'Embeded',
            )

            if _status:
                c2.subheader('')
                if is_delete:
                    dump_json(f'{task_path}/model_embeded_params.json', _dic)
                    _embeded_model_params = _dic
                    c2.success('Selected embeded model removed from this model.')
                else:
                    if not is_training and c2.button('save', key='embeded_model_params'):
                        dump_json(f'{task_path}/model_embeded_params.json', _dic)
                        _embeded_model_params = _dic
                        c2.success(_msg)
            else:
                if _msg:
                    c2.warning(_msg)
        st.write(_embeded_model_params)

    if _choice == 'Model Layers':
        _base_model_params = get_dic_from_json(f'{task_path}/model_bases_params.json')
        _embeded_model_params = get_dic_from_json(f'{task_path}/model_embeded_params.json')
        _model_layer_params = get_dic_from_json(f'{task_path}/model_layer_params.json')

        _dic_inputs_types = {
            'Pretrained': _base_model_params,
            'Embeded': _embeded_model_params,
            'Layer': _model_layer_params,
        }

        _options, _dic = None, {}
        if _model_layer_params:
            _options = list(_model_layer_params.keys())
            _dic = {k: v for k, v in _model_layer_params.items()}
        _key = ''

        _action = c2.radio('', ['New', 'Edit', 'Remove'])
        if _action == 'New':
            _status, _dic = add_layer(c2, st, _dic_inputs_types, _model_layer_params, DIC_Layers)
            if _status:
                c2.subheader('')
                if not is_training and c2.button('save', key='model_layer_params'):
                    dump_json(f'{task_path}/model_layer_params.json', _dic)
                    _model_layer_params = _dic
                    c2.success('Params of model layer saved.')
            else:
                c2.warning(_dic)

        if _action == 'Edit':
            if not _model_layer_params:
                c2.warning('Layers not found.')
            else:
                _options = list(_model_layer_params.keys())

                if _options:
                    _key = c2.selectbox('Select name to edit:', _options, 0)
                    _dic = layer_conf(c2, st, _dic, _dic_inputs_types, _key, _model_layer_params, DIC_Layers, _options=_options)

                    c2.subheader('')
                    if not is_training and c2.button('save', key='model_layer_params'):
                        dump_json(f'{task_path}/model_layer_params.json', _dic)
                        _model_layer_params = _dic
                        c2.success('Params of model layer saved.')

        if _action == 'Remove':
            if not _model_layer_params:
                c2.warning('Layers not found.')
            else:
                _options = list(_model_layer_params.keys())
                _dic = {k: v for k, v in _model_layer_params.items()}

                if _options:
                    _key = c2.selectbox('Select name to edit:', _options, 0)

                if c2.button('remove'):
                    del _dic[_key]
                    dump_json(f'{task_path}/model_layer_params.json', _dic)
                    _model_layer_params = _dic
                    c2.success('Selected model layer removed from this model.')

        st.write(_model_layer_params)

    if _choice == 'Model Inputs':
        _base_model_params = get_dic_from_json(f'{task_path}/model_bases_params.json')
        _embeded_model_params = get_dic_from_json(f'{task_path}/model_embeded_params.json')
        _model_layer_params = get_dic_from_json(f'{task_path}/model_layer_params.json')
        _model_inputs_params = get_dic_from_json(f'{task_path}/model_inputs_params.json')

        _dic_inputs_types = {
            'Pretrained': _base_model_params,
            'Embeded': _embeded_model_params,
            'Layer': _model_layer_params,
        }

        if _model_inputs_params:
            _default = len(_model_inputs_params)
        else:
            _default, _model_inputs_params = 0, []

        ls = inputs_conf(c2, st, _default, _model_inputs_params, _dic_inputs_types)
        if len(ls):
            c2.subheader('')
            if not is_training and c2.button('save', key='model_inputs_params'):
                dump_json(f'{task_path}/model_inputs_params.json', ls)
                _model_inputs_params = ls
                c2.success('Model inputs params saved.')

        st.write(_model_inputs_params)

    if _choice == 'Model Outputs':
        _base_model_params = get_dic_from_json(f'{task_path}/model_bases_params.json')
        _embeded_model_params = get_dic_from_json(f'{task_path}/model_embeded_params.json')
        _model_layer_params = get_dic_from_json(f'{task_path}/model_layer_params.json')
        _model_outputs_params = get_dic_from_json(f'{task_path}/model_outputs_params.json')

        _dic_outputs_types = {
            'Pretrained': _base_model_params,
            'Embeded': _embeded_model_params,
            'Layer': _model_layer_params,
        }

        _options, _dic = None, {}
        if _model_outputs_params:
            _options = list(_model_outputs_params.keys())
            _dic = {k: v for k, v in _model_outputs_params.items()}
        else:
            _model_outputs_params = {}
        _key = ''

        _action = c2.radio('', ['New', 'Edit', 'Remove'])
        if _action == 'New':
            _status, _dic = add_output(c2, _dic_outputs_types, _model_outputs_params, DIC_Losses, DIC_Metrics)
            if _status:
                c2.subheader('')
                if not is_training and c2.button('save', key='model_output_params'):
                    dump_json(f'{task_path}/model_outputs_params.json', _dic)
                    _model_outputs_params = _dic
                    c2.success('Params of model output saved.')
            else:
                c2.warning(_dic)

        if _action == 'Edit':
            if not _model_outputs_params:
                c2.warning('Outputs not found.')
            else:
                _options = list(_model_outputs_params.keys())
                _key = c2.selectbox('Select name to edit:', _options, 0)
                _dic = output_conf(c2, _dic, _dic_outputs_types, _key, _model_outputs_params, DIC_Losses, DIC_Metrics)

                c2.subheader('')
                if not is_training and c2.button('save', key='model_outputs_params'):
                    dump_json(f'{task_path}/model_outputs_params.json', _dic)
                    _model_outputs_params = _dic
                    c2.success('Params of model output saved.')

        if _action == 'Remove':
            if not _model_outputs_params:
                c2.warning('Outputs not found.')
            else:
                _options = list(_model_outputs_params.keys())
                _dic = {k: v for k, v in _model_outputs_params.items()}

                if _options:
                    _key = c2.selectbox('Select an output:', _options, 0)

                if c2.button('remove'):
                    del _dic[_key]
                    dump_json(f'{task_path}/model_outputs_params.json', _dic)
                    _model_outputs_params = _dic
                    c2.success('Selected model output removed from this model.')

        st.write(_model_outputs_params)

    if _choice == 'Model Optimizer':
        _model_optimizer_params = get_dic_from_json(f'{task_path}/model_optimizer_params.json')

        _options = list(DIC_Optimizers.keys())
        _default = _options[0]
        _params = None
        if _model_optimizer_params and len(_model_optimizer_params) > 0:
            _default = _model_optimizer_params.get('func')
            _params = _model_optimizer_params.get('params')
        _index = _options.index(_default)

        _dic = {}
        c1, c2 = st.beta_columns([1, 3])
        _option = c1.selectbox('Select an optimizer:', _options, index=_index)
        _dic['func'] = _option
        if not _params:
            _params = DIC_Optimizers.get(_option).get('params')
        if _params:
            _params = eval(c2.text_input(f'{_option} params:', _params))
            _dic['params'] = _params
        else:
            c2.write(f'{_option}: None params')

        _default = _model_optimizer_params.get('loss_weights', []) if _model_optimizer_params else [1, 1]
        _loss_weights = st.text_input('model loss weights:', _default).strip()

        _dic['loss_weights'] = None if not len(_loss_weights) or _loss_weights == 'None' else eval(_loss_weights)

        if st.button('save'):
            dump_json(f'{task_path}/model_optimizer_params.json', _dic)
            _model_optimizer_params = _dic
            st.success('Model optimizer saved.')

        st.write(_model_optimizer_params)


def training_data_params(task_path, is_training=False):
    _training_data_params = get_dic_from_json(f'{task_path}/params_data.json')
    _dic_data = {}

    st.title('Training data params')
    with st.beta_expander("Data loader params"):
        _dir_data = 'data'

        _options, _default, _params = get_default_params(
            _training_data_params, DIC_DataLoaders, 'data_loader_params')

        col1, col2 = st.beta_columns(2)
        _index = _options.index(_default)
        _option = col1.selectbox('Select a data loader', _options, index=_index)

        if not _params:
            _params = DIC_DataLoaders.get(_option).get('params')
        _params_new = {}
        if _params:
            _params_new['dir_data'] = _dir_data

            _options = tuple(os.walk(f'{_dir_data}/'))[0][1]
            _default = _options.index(_params.get('data_cls')) if _params.get('data_cls') else 0
            _params_new['data_cls'] = col2.selectbox('Select a data set', _options, _default).split('/')[-1]

            if 't1' in _params:
                _params_new['t1'] = st.text_input('Column name of first text', _params.get('t1'))
            if 't2' in _params:
                _params_new['t2'] = st.text_input('Column name of second text', _params.get('t2'))
            if 'label' in _params:
                _params_new['label'] = st.text_input('Column name of sentence label', _params.get('label'))
            DIC_DataLoaders[_option]['params'] = _params_new
            _dic_data['data_loader_params'] = {'func': _option, 'params': _params_new}
        else:
            col2.write(f'{_option}: None params')
            _dic_data['data_loader_params'] = {'func': _option}

    _data_cls = _params_new.get("data_cls")
    if _data_cls:
        st.subheader('')
        with st.beta_expander("Training data set params"):
            _options = glob.glob(f'{_dir_data}/{_data_cls}/*.*')

            _default = get_default_input(None, _training_data_params, 'fns_train')

            if _default and not set(_options).intersection(set(_default)):
                _default = None
            _dic_data = multi_options(task_path, 'fns_train', _options, _dic_data,
                                      _default=_default, is_set=True, is_params=False)

            _default = get_default_input(None, _training_data_params, 'fns_dev')
            if _default and not set(_options).intersection(set(_default)):
                _default = None
            _dic_data = multi_options(task_path, 'fns_dev', _options, _dic_data,
                                      _default=_default, is_set=True, is_params=False)

            _default = get_default_input(None, _training_data_params, 'fns_test')
            if _default and not set(_options).intersection(set(_default)):
                _default = None
            _dic_data = multi_options(task_path, 'fns_test', _options, _dic_data,
                                      _default=_default, is_set=True, is_params=False)

    st.subheader('')
    with st.beta_expander("Data generator params"):
        _options, _default = get_default(
            _training_data_params, DIC_Generators_for_train, 'data_generator_for_train', is_num=True)
        _dic_data['data_generator_for_train'] = st.selectbox('data generator for train', _options, index=_default)
        # _dic_data['data_generator_for_train'] = 'data_generator_for_train'

        # _options, _default = get_default(
        #     _training_data_params, DIC_Generators_for_pred, 'data_generator_for_pred', is_num=True)
        # _dic_data['data_generator_for_pred'] = st.selectbox('data_generator_for_pred', _options, index=_default)
        _dic_data['data_generator_for_pred'] = 'data_generator_for_pred'

    st.subheader('')
    with st.beta_expander("Other params"):
        col1, col2, col3 = st.beta_columns([1, 1, 2])

        _default = get_default_input(64, _training_data_params, 'batch_size')
        _dic_data['batch_size'] = col1.number_input(
            'training batch size:', min_value=16, max_value=512, value=_default, step=16)

        _default = get_default_input('sigmoid', _training_data_params, 'activation')
        _dic_data['activation'] = col2.text_input('output dense activation:', _default)

        _default = get_default_input('', _training_data_params, 'fn_labeler')
        _fn_labeler = col3.text_input('labeler name(should be a pickle file):', _default).strip()

        if not len(_fn_labeler) or _fn_labeler == 'None':
            _fn_labeler = None
        _dic_data['fn_labeler'] = _fn_labeler

        _default = get_default_input(0, _training_data_params, 'is_sequence')
        _dic_data['is_sequence'] = st.radio('is sequence task?', [True, False], index=[True, False].index(_default))

    st.subheader('')
    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_data.json', _dic_data)
        st.success('Params saved.')


def training_params(task_path, is_training=False):
    _training_params = get_dic_from_json(f'{task_path}/params_train.json')
    _dic_train = {}

    st.title('Training params')
    col1, col2 = st.beta_columns([1, 3])
    _default = get_default_input(30, _training_params, 'epochs')
    _dic_train['epochs'] = col1.number_input(
        'training epochs:', min_value=10, max_value=1000, value=_default, step=10)

    _default = get_default_input('', _training_params, 'checkpoint')
    _checkpoint = col2.text_input('model checkpoint:', _default).strip()
    if not len(_checkpoint) or _checkpoint == 'None':
        _checkpoint = None
    _dic_train['checkpoint'] = _checkpoint

    _dic_train['cp_loss'] = {}
    _dic_train['early_stopping'] = {}

    col1, col2, col3 = st.beta_columns([2, 1, 1])
    _default = get_default_input('val_accuracy', _training_params, 'cp_loss', sub_tag='monitor')
    _monitor = col1.text_input('validate score monitor by:', _default).strip()
    _dic_train['cp_loss']['monitor'] = _monitor
    _dic_train['early_stopping']['monitor'] = _monitor

    _default = get_default_input('max', _training_params, 'cp_loss', sub_tag='mode')
    _mode = col2.text_input('score monitor scale:', _default).strip()
    _dic_train['cp_loss']['mode'] = _mode
    _dic_train['early_stopping']['mode'] = _mode

    _dic_train['cp_loss']['verbose'] = 1
    _dic_train['early_stopping']['verbose'] = 1

    _dic_train['cp_loss']['save_best_only'] = True
    _dic_train['cp_loss']['save_weights_only'] = True

    _default = get_default_input(5, _training_params, 'early_stopping', sub_tag='patience')
    _patience = col3.number_input('early stop patience:',
                                  min_value=0, max_value=100, value=_default, step=5)
    _dic_train['early_stopping']['patience'] = _patience

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_train.json', _dic_train)
        st.success('Params saved.')


def predict_params(task_path, is_training=False):
    _predict_params = get_dic_from_json(f'{task_path}/params_pred.json')
    _dic_pred = {}

    titles = [st.title('Predict params')]

    _options = list(DIC_Resolvers.keys())
    _default = _options.index(_predict_params.get('resolver')) if _predict_params and 'resolver' in _predict_params else 0

    c1, c2 = st.beta_columns(2)
    _dic_pred['resolver'] = c1.selectbox('Select a resolver:', _options, index=_default)

    _default = get_default_input(None, _predict_params, 'threshold')
    _dic_pred['threshold'] = c2.text_input('Score metric threshold:', _default).strip()

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_pred.json', _dic_pred)
        st.success('Params saved.')
