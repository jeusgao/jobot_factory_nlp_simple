#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-14 11:47:56
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import shutil
import streamlit as st

from init_params import env_init
from utils import dump_json
from gui import (
    model_params,
    training_data_params,
    training_params,
    predict_params,
    TrainingGUI,
)

env_init()


def _create_task():
    st.title('Create a new task')
    task_path = st.empty()
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


def _task_management(task_path):
    task_path = f"hub/models/{task_path}"
    action = st.sidebar.radio(
        'Actions of task',
        [
            'Training params configuration',
            'Train the model',
            'Evaluate the model',
            f'Remove task - {task_path}',
        ]
    )

    is_running = os.path.exists(f'{task_path}/state.json')

    if action == 'Training params configuration':
        cfg = st.radio('Select a params set to customize:', [
                       'Training data', 'Model', 'Taining task', 'Predictor params'])
        if cfg == 'Training data':
            training_data_params(task_path, is_training=is_running)
        if cfg == 'Model':
            model_params(task_path, is_training=is_running)
        if cfg == 'Taining task':
            training_params(task_path, is_training=is_running)
        if cfg == 'Predictor params':
            predict_params(task_path, is_training=is_running)

    if action == 'Train the model':
        training_gui = TrainingGUI(task_path=task_path, is_running=is_running)
        training_gui.train()

    if action == 'Evaluate the model':
        if os.path.exists(f'{task_path}/model.h5'):
            training_gui = TrainingGUI(task_path=task_path, is_running=is_running, is_eval=True)
            training_gui.train()
        else:
            st.warning('Model weights not found, please train the model first.')

    if action == f'Remove task - {task_path}':
        if is_running:
            st.warning('Cannot remove task when task training is running ...')
        else:
            block_title = st.title(f'Are you sure remove task {task_path}?')
            block_remove = st.empty()
            block_cancel = st.empty()
            _remove_task(task_path, block_title, block_remove, block_cancel)


task_action = st.sidebar.radio('', ['New task', 'Select task from exists'])
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
