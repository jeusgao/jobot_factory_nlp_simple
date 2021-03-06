#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-14 11:47:56
# @Author  : Joe Gao (jeusgao@163.com)

import os
import re
import json
import shutil
import streamlit as st

from utils import dump_json
from gui import (
    model_params,
    training_data_params,
    training_params,
    predict_params,
    TrainingGUI,
)


def _regex_test():
    _text = st.text_input('Input a text please', '')
    _reg = st.text_input('Input a regex string please', '')
    _mode = st.radio('Select a reg mode please', ['Match', 'Split'])
    if st.button('Submit'):
        st.write(re.match(_reg, _text) if _mode == 'Match' else re.split(_reg, _text))


def _create_task():
    st.title('Create a new task')
    _name = st.text_input('Input a task name please', '')
    if st.button('Submit'):
        _name = _name.strip()
        if _name:
            _name = f"hub/models/{_name}"
            if not os.path.exists(_name):
                os.makedirs(_name)
                st.success(f'Task {_name} created.')
            else:
                st.warning(f'Task {_name} already existed.')
        else:
            st.warning('Please input a Task Name.')


def _remove_task(task_path, block_title, block_remove, block_cancel):
    _confirm = block_remove.button('Confirm')
    _cancel = block_cancel.button('Cancel')
    if _confirm:
        shutil.rmtree(task_path)
        block_remove.empty()
        block_cancel.empty()
        block_title.success(f'Task {task_path} removed.')
    if _cancel:
        block_remove.empty()
        block_cancel.empty()
        block_title.success(f'Task remove canceled.')


def _duplicate_task(task_path, new_name, block_ok):
    st.title('Duplicate task ...')
    _name = st.text_input('Input a new task name please', '')
    if st.button('Submit'):
        _name = _name.strip()
        if _name:
            if not os.path.exists(f'hub/models/{_name}'):
                os.makedirs(f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_bases_params.json'):
                    shutil.copy(f'{task_path}/model_bases_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_common_params.json'):
                    shutil.copy(f'{task_path}/model_common_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_embeded_params.json'):
                    shutil.copy(f'{task_path}/model_embeded_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_inputs_params.json'):
                    shutil.copy(f'{task_path}/model_inputs_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_layer_params.json'):
                    shutil.copy(f'{task_path}/model_layer_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_optimizer_params.json'):
                    shutil.copy(f'{task_path}/model_optimizer_params.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/model_outputs_params.json'):
                    shutil.copy(f'{task_path}/model_outputs_params.json', f'hub/models/{_name}')

                if os.path.exists(f'{task_path}/params_data.json'):
                    shutil.copy(f'{task_path}/params_data.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/params_train.json'):
                    shutil.copy(f'{task_path}/params_train.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/params_pred.json'):
                    shutil.copy(f'{task_path}/params_pred.json', f'hub/models/{_name}')
                if os.path.exists(f'{task_path}/tmp_layers.txt'):
                    shutil.copy(f'{task_path}/tmp_layers.txt', f'hub/models/{_name}')

                st.success(f'Task {_name} created.')
            else:
                st.warning(f'Task {_name} already existed.')
        else:
            st.warning('Please input a Task Name.')


def _task_management(task_path):
    task_path = f"hub/models/{task_path}"
    action = st.sidebar.radio(
        'Actions of task',
        [
            'Task params configuration',
            'Train the model of task',
            'Evaluate the task',
            f'Duplicate task - {task_path}',
            f'Remove task - {task_path}',
        ]
    )

    is_running = os.path.exists(f'{task_path}/state.json')

    if action == 'Task params configuration':
        st.subheader('Select a params set to customize:')

        c1, c2, c3, c4 = st.beta_columns(4)

        def _get_cur():
            if os.path.exists('_tmp_param_status'):
                with open('_tmp_param_status', 'r') as f:
                    _cur = json.load(f).get('_cur', 0)
                return _cur
            return 0

        _cur = _get_cur()

        if c1.button('Training data', key='btn_1'):
            with open('_tmp_param_status', 'w') as f:
                json.dump({'_cur': 1}, f)
            _cur = _get_cur()
        if c2.button('Model params', key='btn_2'):
            with open('_tmp_param_status', 'w') as f:
                json.dump({'_cur': 2}, f)
            _cur = _get_cur()
        if c3.button('Taining task', key='btn_3'):
            with open('_tmp_param_status', 'w') as f:
                json.dump({'_cur': 3}, f)
            _cur = _get_cur()
        if c4.button('Predictor params', key='btn_4'):
            with open('_tmp_param_status', 'w') as f:
                json.dump({'_cur': 4}, f)
            _cur = _get_cur()

        if _cur > 0:
            if _cur == 1:
                training_data_params(task_path, is_training=is_running)
            if _cur == 2:
                model_params(task_path, is_training=is_running)
            if _cur == 3:
                training_params(task_path, is_training=is_running)
            if _cur == 4:
                predict_params(task_path, is_training=is_running)

    if action == 'Train the model of task':
        training_gui = TrainingGUI(task_path=task_path, is_running=is_running)
        training_gui.train()

    if action == 'Evaluate the task':
        if os.path.exists(f'{task_path}/model.h5'):
            training_gui = TrainingGUI(task_path=task_path, is_running=is_running, is_eval=True)
            training_gui.train()
        else:
            st.warning('Model weights not found, please train the model first.')

    if action == f'Duplicate task - {task_path}':
        new_name = st.empty()
        block_ok = st.empty()
        _duplicate_task(task_path, new_name, block_ok)

    if action == f'Remove task - {task_path}':
        if is_running:
            st.warning('Cannot remove task when task training is running ...')
        else:
            block_title = st.title(f'Are you sure remove task {task_path}?')
            block_remove = st.empty()
            block_cancel = st.empty()
            _remove_task(task_path, block_title, block_remove, block_cancel)


# from init_params import env_init
# if st.sidebar.button('Init params templates'):
#     env_init()
#     st.success('Params templates initialized.')

task_action = st.sidebar.radio('', ['New task', 'Select task from exists', 'Regex test', 'KE'])

if task_action == 'New task':
    _create_task()

if task_action == 'Select task from exists':
    _options = tuple(os.walk('hub/models'))[0][1]
    if _options:
        task_path = st.sidebar.selectbox('Select a task name', _options)
        if task_path:
            _task_management(task_path)
    else:
        st.warning('No task exists, please create a task first.')

if task_action == 'Regex test':
    _regex_test()
