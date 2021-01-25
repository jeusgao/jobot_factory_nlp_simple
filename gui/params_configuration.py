#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-13 13:00:47
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import glob
import streamlit as st

from utils import get_dic_from_json, dump_json
from .components import (
    get_default_input,
    get_default,
    get_default_params,
    single_option,
    multi_options,
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

fn_tmp_layers = 'tmp_layers.txt'
fn_tmp_metrics = 'tmp_metrics.txt'


def model_params(task_path, is_training=False):
    _model_params = get_dic_from_json(f'{task_path}/params_model.json')
    _dic = {}

    titles = [st.title('Model params'), st.info('Base params'), st.subheader('Learning rate')]

    # _dic['LR'] = float(
    #     st.text_input('', _model_params.get('LR') if _model_params else '2e-5')
    # )

    st.subheader('Max lenth')
    maxlen = st.number_input(
        'input max lenth',
        min_value=8,
        max_value=512,
        value=_model_params.get('maxlen') if _model_params else 32,
        step=8,
    )
    _dic['maxlen'] = maxlen

    _is_pair = True
    if _model_params and _model_params.get('ML') == _model_params.get('maxlen'):
        _is_pair = False

    is_pair = st.radio('is pair inputs', [True, False], [True, False].index(_is_pair))
    _dic['ML'] = maxlen * 2 + 3 if is_pair else maxlen

    st.subheader('Pretrained base model')
    _options, _default = get_default(_model_params, DIC_Bases, 'base_code')
    _dic['base_code'] = st.selectbox('base model', _options, _options.index(_default))

    _options = tuple(os.walk('hub/bases'))[0][1]
    _default = _options[0]
    if _model_params:
        _default = _model_params.get('base_params').get('fn_config').split('/')[-2]
    _base_path = st.selectbox('base', _options, _options.index(_default))
    _dic['base_params'] = {
        'fn_config': f'hub/bases/{_base_path}/bert_config.json',
        'fn_base_model': f'hub/bases/{_base_path}/bert_model.ckpt',
        'training': st.radio('is base training', [True, False], index=1),
        'trainable': st.radio('is base trainable', [True, False], index=0),
    }

    st.subheader('Tokenizer')
    _options, _default = get_default(_model_params, DIC_Tokenizers, 'tokenizer_code')
    _dic['tokenizer_code'] = st.selectbox('tokenizer', _options, _options.index(_default))
    _dic['tokenizer_params'] = {'fn_vocab': f'hub/bases/{_base_path}/vocab.txt'}

    st.info('Model params:')

    _dic_model = {}

    st.subheader('Optimizer')
    _options, _default, _params = get_default_params(
        _model_params, DIC_Optimizers, 'model_params', sub_tag='optimizer')
    _dic_model = single_option(
        'optimizer', DIC_Optimizers, _dic_model, _options,
        _params=_params, _index=_options.index(_default)
    )

    st.subheader('Loss')
    _options, _default, _params = get_default_params(
        _model_params, DIC_Losses, 'model_params', sub_tag='loss')
    _dic_model = single_option(
        'loss', DIC_Losses, _dic_model, _options,
        _params=_params, _index=_options.index(_default)
    )

    st.subheader('Metrics:')
    _default = None
    if _model_params:
        _default = [m.get('func') for m in _model_params.get('model_params').get('metrics')]
    _dic_model = multi_options(
        task_path, 'metrics', list(DIC_Metrics.keys()), _dic_model,
        _default=_default, is_set=True, tpl_dic=DIC_Metrics)

    st.subheader('Layers:')
    _dic_model = multi_options(
        task_path, 'layers', list(DIC_Layers.keys()), _dic_model,
        fn=fn_tmp_layers, tpl_dic=DIC_Layers
    )

    _dic['model_params'] = _dic_model

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_model.json', _dic)
        st.success('Params saved.')


def training_data_params(task_path, is_training=False):
    _training_data_params = get_dic_from_json(f'{task_path}/params_data.json')
    _dic_data = {}

    titles = [st.title('Training data params'), st.info('Data loader params')]

    st.write('Select a data loader')
    _options, _default, _params = get_default_params(
        _training_data_params, DIC_DataLoaders, 'data_loader_params')

    _index = _options.index(_default)
    _option = st.selectbox('', _options, index=_index)
    if not _params:
        _params = DIC_DataLoaders.get(_option).get('params')
    if _params:
        _params_new = {}
        _params_new['data'] = 'data'

        _options = tuple(os.walk('data/'))[0][1]
        _default = _options.index(_params.get('data_cls')) if _params.get('data_cls') else 0
        _params_new['data_cls'] = st.selectbox('Select a data set', _options, _default).split('/')[-1]

        if 't1' in _params:
            _params_new['t1'] = st.text_input('Column name of first text', _params.get('t1'))
        if 't2' in _params:
            _params_new['t2'] = st.text_input('Column name of second text', _params.get('t2'))
        if 'label' in _params:
            _params_new['label'] = st.text_input('Column name of sentence label', _params.get('label'))
        DIC_DataLoaders[_option]['params'] = _params_new
        _dic_data['data_loader_params'] = {'func': _option, 'params': _params_new}
    else:
        st.write(f'{_option}: None params')
        _dic_data['data_loader_params'] = {'func': _option}

    # _dic_data = single_option(
    #     'data_loader_params', DIC_DataLoaders, _dic_data, _options,
    #     _params=_params, _index=_options.index(_default))

    # _dir_data = _dic_data.get("data_loader_params").get("params").get("dir_data")
    _dir_data = 'data'
    _data_cls = _dic_data.get("data_loader_params").get("params").get("data_cls")

    if _dir_data and _data_cls:
        st.info('Training data set params')
        _options = glob.glob(f'{_dir_data}/{_data_cls}/*.*')

        _default = get_default_input(None, _training_data_params, 'fns_train')
        _dic_data = multi_options(task_path, 'fns_train', _options, _dic_data,
                                  _default=_default, is_set=True, is_params=False)

        _default = get_default_input(None, _training_data_params, 'fns_dev')
        _dic_data = multi_options(task_path, 'fns_dev', _options, _dic_data,
                                  _default=_default, is_set=True, is_params=False)

        _default = get_default_input(None, _training_data_params, 'fns_test')
        _dic_data = multi_options(task_path, 'fns_test', _options, _dic_data,
                                  _default=_default, is_set=True, is_params=False)

    # st.info('Data generator params')
    # _options, _default = get_default(
    #     _training_data_params, DIC_Generators_for_train, 'data_generator_for_train', is_num=True)
    # _dic_data['data_generator_for_train'] = st.selectbox('data_generator_for_train', _options, index=_default)

    # _options, _default = get_default(
    #     _training_data_params, DIC_Generators_for_pred, 'data_generator_for_pred', is_num=True)
    # _dic_data['data_generator_for_pred'] = st.selectbox('data_generator_for_pred', _options, index=_default)

    _dic_data['data_generator_for_train'] = 'data_generator_for_train'
    _dic_data['data_generator_for_pred'] = 'data_generator_for_pred'

    st.info('...')
    _default = get_default_input(64, _training_data_params, 'batch_size')
    _dic_data['batch_size'] = st.number_input(
        'training batch size:', min_value=16, max_value=512, value=_default, step=16)

    _default = get_default_input('sigmoid', _training_data_params, 'activation')
    _dic_data['activation'] = st.text_input('output dense activation:', _default)

    _default = get_default_input('', _training_data_params, 'fn_labeler')
    _fn_labeler = st.text_input('labeler filename(should be a pickle file):', _default).strip()

    if not len(_fn_labeler) or _fn_labeler == 'None':
        _fn_labeler = None
    _dic_data['fn_labeler'] = _fn_labeler

    _default = get_default_input(0, _training_data_params, 'is_sequence')
    _dic_data['is_sequence'] = st.radio('is sequence task?', [True, False], index=[True, False].index(_default))

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_data.json', _dic_data)
        st.success('Params saved.')


def training_params(task_path, is_training=False):
    _training_params = get_dic_from_json(f'{task_path}/params_train.json')
    _dic_train = {}

    st.title('Training params')
    _default = get_default_input(30, _training_params, 'epochs')
    _dic_train['epochs'] = st.number_input(
        'training epochs:', min_value=10, max_value=1000, value=_default, step=10)

    _default = get_default_input('', _training_params, 'checkpoint')
    _checkpoint = st.text_input('model checkpoint:', _default).strip()
    if not len(_checkpoint) or _checkpoint == 'None':
        _checkpoint = None
    _dic_train['checkpoint'] = _checkpoint

    _dic_train['cp_loss'] = {}
    _dic_train['early_stopping'] = {}

    _default = get_default_input('val_accuracy', _training_params, 'cp_loss', sub_tag='monitor')
    _monitor = st.text_input('validate score monitor by:', _default).strip()
    _dic_train['cp_loss']['monitor'] = _monitor
    _dic_train['early_stopping']['monitor'] = _monitor

    _default = get_default_input('max', _training_params, 'cp_loss', sub_tag='mode')
    _mode = st.text_input('validate score monitor scale:', _default).strip()
    _dic_train['cp_loss']['mode'] = _mode
    _dic_train['early_stopping']['mode'] = _mode

    _dic_train['cp_loss']['verbose'] = 1
    _dic_train['early_stopping']['verbose'] = 1

    _dic_train['cp_loss']['save_best_only'] = True
    _dic_train['cp_loss']['save_weights_only'] = True

    _default = get_default_input(5, _training_params, 'early_stopping', sub_tag='patience')
    _patience = st.number_input('patience for early stop training:',
                                min_value=0, max_value=100, value=_default, step=5)
    _dic_train['early_stopping']['patience'] = _patience

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_train.json', _dic_train)
        st.success('Params saved.')


def predict_params(task_path, is_training=False):
    _predict_params = get_dic_from_json(f'{task_path}/params_pred.json')
    _dic_pred = {}

    titles = [st.title('Predict params')]

    _default = get_default_input(None, _predict_params, 'threshold')
    _dic_pred['threshold'] = st.text_input('Score metric threshold:', _default).strip()

    _dic_pred['resolver'] = 'resolver'

    if not is_training and st.button('save'):
        dump_json(f'{task_path}/params_pred.json', _dic_pred)
        st.success('Params saved.')
