#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-14 11:47:56
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import streamlit as st

from utils import dump_json
from gui import (
    model_params,
    training_data_params,
    training_params,
    TrainingGUI,
)

task_path = None
fn_model = None

task_action = st.sidebar.radio('', ['New task', 'Select task from exists'])
if task_action == 'New task':
    task_path = st.sidebar.text_input('Input a task name', '')
    if task_path:
        task_path = f"hub/models/{task_path}"
        if not os.path.exists(task_path):
            os.makedirs(task_path)

if task_action == 'Select task from exists':
    _options = tuple(os.walk('hub/models'))[0][1]
    task_path = st.sidebar.selectbox('Select a task name', _options)
    if task_path:
        task_path = f"hub/models/{task_path}"

if task_path:
    action = st.sidebar.radio(
        'Actions of task',
        [
            'Training params configuration',
            'Train the model',
            'Evaluate the model',
            'Predictor params',
        ]
    )

    step = None
    is_running = os.path.exists(f'{task_path}/state.json')

    if action == 'Training params configuration':
        cfg = st.radio('Select a params set to customize:', ['Training data', 'Model', 'Taining task'])
        if cfg == 'Training data':
            training_data_params(task_path, is_training=is_running)
        if cfg == 'Model':
            model_params(task_path, is_training=is_running)
        if cfg == 'Taining task':
            training_params(task_path, is_training=is_running)

    if action == 'Train the model':
        training_gui = TrainingGUI(task_path=task_path, is_running=is_running)
        training_gui.train()

    if action == 'Evaluate the model':
        if os.path.exists(f'{task_path}/model.h5'):
            training_gui = TrainingGUI(task_path=task_path, is_running=is_running, is_eval=True)
            training_gui.train()
        else:
            st.warning('Model weights not found, please train the model first.')

    if action == 'Predictor params':
        dump_json(
            f'{task_path}/params_pred.json',
            {
                'data_util': 'data_generator_pred',
                'fn_weights': f'{task_path}/model.h5',
            }
        )