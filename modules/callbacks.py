#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-05 14:57:15
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import time
from backend import keras


class TrainingCallbacks(keras.callbacks.Callback):
    def __init__(self, task_path='', log_name='training'):
        self.task_path = task_path
        self.log_name = log_name

    def on_train_begin(self, logs):
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'w') as f:
            f.write('')
        # if os.path.exists(f'{self.task_path}/{self.log_name}_logs.json'):
            # os.remove(f'{self.task_path}/{self.log_name}_logs.json')

    def on_train_end(self, logs):
        txt = json.dumps({
            'EPOCH': 'Finished',
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            'scores': logs
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'{txt}\n')

        os.remove(f'{self.task_path}/state.json')

    def on_epoch_begin(self, epoch, logs):
        txt = json.dumps({
            'EPOCH': epoch,
            'state': 'Begin',
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'{txt}\n')

    def on_epoch_end(self, epoch, logs):
        txt = json.dumps({
            'EPOCH': epoch,
            'state': 'end',
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            'scores': str(logs)
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'{txt}\n')

    def on_train_batch_end(self, epoch, logs):
        txt = json.dumps({
            'batch': epoch,
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            'scores': str(logs)
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'\t{txt}\n')


class EvaluatingCallbacks(keras.callbacks.Callback):
    def __init__(self, task_path='', log_name='evaluating'):
        self.task_path = task_path
        self.log_name = log_name

    def on_test_begin(self, logs=None):
        print('Evaluating ...')
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'w') as f:
            f.write('')
        # if os.path.exists(f'{self.task_path}/{self.log_name}_logs.json'):
        #     os.remove(f'{self.task_path}/{self.log_name}_logs.json')

    def on_test_end(self, logs=None):
        txt = json.dumps({
            'EPOCH': 'Finished',
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            'scores': logs
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'{txt}\n')
        os.remove(f'{self.task_path}/state.json')

    def on_test_batch_end(self, batch, logs=None):
        txt = json.dumps({
            'batch': batch,
            'time': time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
            'scores': logs
        })
        with open(f'{self.task_path}/{self.log_name}_logs.json', 'a') as f:
            f.write(f'\t{txt}\n')
