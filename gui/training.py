#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-15 12:35:35
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import time
import pandas as pd
import streamlit as st
from celery.app.control import Control

from utils import (
    get_params,
    get_lines,
    get_key_from_json,
    kill_process,
    dump_json,
)
from celery_training import training_excute, app

celery_control = Control(app=app)


class TrainingGUI(object):
    def __init__(self, task_path, is_running=False, is_eval=False):
        self.fn_state = f'{task_path}/state.json'
        self.is_running = is_running
        self._action = 'evaluating' if is_eval else 'training'
        self.fn_model = f'{task_path}/model.h5'
        self.is_configed = all([
            os.path.exists(f'{task_path}/params_model.json'),
            os.path.exists(f'{task_path}/params_data.json'),
            os.path.exists(f'{task_path}/params_train.json'),
            os.path.exists(f'{task_path}/params_pred.json'),
        ])
        self.task_path = task_path
        self.is_eval = is_eval
        self.is_model_structure_showed = False

        st.title(f'Task {task_path.split("/")[-1]} {self._action}...')

    def _print_logs(self, _n_curr=0, state='begin', _freq=100, train_graph=None, valid_graph=None, is_running=True, log_epoch=None, logs_graph=None):
        logs = get_lines(f'{self.task_path}/{self._action}_logs.json')
        for i, log in enumerate(logs[_n_curr:]):
            if 'EPOCH' in log:
                log = json.loads(log.strip())
                state = log.get('EPOCH')
                if 'scores' in log:
                    if not self.is_eval:
                        scores = {
                            k: v
                            for k, v in log.get('scores').items() if k.startswith('val_')
                        }
                        valid_graph.add_rows(pd.json_normalize([scores]))
                        st.text(f'EPOCH: {state}, Scores: {scores}')
                if log_epoch:
                    log_epoch.json(log)
            else:
                try:
                    log = json.loads(log.strip())
                    if self.is_eval:
                        if is_running:
                            train_graph.json(log)
                    else:
                        if log.get('batch') % _freq == 0:
                            train_graph.add_rows(pd.json_normalize([log.get('scores')]))
                except:
                    continue
            logs_graph.json(log)
        _n_curr = len(logs)
        return _n_curr, state, train_graph

    def _show_structure(self):
        if not self.is_model_structure_showed:
            st.info(f'Model layers')
            st.json(get_lines(f'{self.task_path}/model_structs.txt'))
            self.is_model_structure_showed = True

    def _monitoring(self):
        _block = st.empty()
        _start = _block.button('Start monitor', key=f'{self._action}_button')
        _freq = st.number_input('Lines per update:', min_value=5, max_value=1000,
                                value=10, step=5, key=f'{self._action}_frequecy')
        if _start:
            _stop = _block.button('Stop monitor')
            st.info(f'{self._action.capitalize()} logs')

            train_graph = st.json('') if self.is_eval else st.line_chart()
            valid_graph = st.empty() if self.is_eval else st.line_chart()
            log_epoch = st.empty()
            logs_graph = st.empty()

            self._show_structure()

            state = 'begin'
            _n_curr = 0
            while not state == 'Finished':
                _n_curr, state, _ = self._print_logs(
                    _n_curr=_n_curr,
                    state=state,
                    _freq=_freq,
                    train_graph=train_graph,
                    valid_graph=valid_graph,
                    log_epoch=log_epoch,
                    logs_graph=logs_graph,
                )
            log_epoch.empty()
            if self.is_eval:
                train_graph.empty()

            st.success(f'{self._action.capitalize()} accomplished.')

        self._show_structure()

    def _start_training(self):
        _block = st.empty()
        _start = _block.button(f'Start {self._action}...')
        if _start:
            dump_json(self.fn_state, {f'{self._action}_state': True})
            res = training_excute.delay(self.task_path, action=self._action)
            dump_json(f'{self.task_path}/training_task.id', {'task_id': res.id})
            _block.empty()
            time.sleep(15)
            _stop = _block.button(f'Stop {self._action}, or think twice before you click me...')
            if _stop:
                _block.empty()
                self._stop_training()
                self._start_training()
            else:
                self._monitoring()
        else:
            if os.path.exists(f'{self.task_path}/{self._action}_logs.json'):
                st.info(f'Last {self._action} logs')
                _n_curr, state, train_graph = self._print_logs(
                    train_graph=st.json('') if self.is_eval else st.line_chart(),
                    valid_graph=st.empty() if self.is_eval else st.line_chart(),
                    logs_graph=st.empty(),
                )
                if self.is_eval:
                    train_graph.empty()

        self._show_structure()

    def _stop_training(self):
        task_id = get_key_from_json(f'{self.task_path}/training_task.id', 'task_id')
        if task_id:
            celery_control.revoke(str(task_id), terminate=True)

            kill_process(f'{self.task_path}/training.pid')

            os.remove(self.fn_state)
            time.sleep(20)
            st.warning(f'{self._action} stopped.')

    def train(self):
        if not self.is_configed:
            st.warning('Task params not found, please customize the params for the task first.')
        else:
            params = {
                'target_path': self.task_path,
                'params_model': get_params(f'{self.task_path}/params_model.json'),
                'params_data': get_params(f'{self.task_path}/params_data.json'),
                'params_train': get_params(f'{self.task_path}/params_train.json'),
            }

            _state = get_key_from_json(self.fn_state, f'{self._action}_state')

            if not self.is_running:
                self._start_training()
            else:
                if _state:
                    _block = st.empty()
                    _stop = _block.button(f'Stop {self._action}, or think twice before you click me...')
                    if _stop:
                        _block.empty()
                        self._stop_training()
                        self._start_training()
                    else:
                        self._monitoring()
                else:
                    _, _, train_graph = self._print_logs(
                        train_graph=st.empty() if self.is_eval else st.line_chart(),
                        valid_graph=st.empty() if self.is_eval else st.line_chart(),
                        is_running=_state,
                        logs_graph=st.empty(),
                    )
                    if self.is_eval:
                        train_graph.empty()

                    self._show_structure()
