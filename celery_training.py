#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-12-13 14:51:55
# @Author  : Joe Gao (jeusgao@163.com)

import os
from celery import Celery

app = Celery(
    'celery_training',
    broker='redis://localhost:6379/1',
    backend='redis',
)


@app.task
def training_excute(task_path, action='training'):
    str = f'python3 trainer.py -p={task_path} -a={action}'
    os.system(str)
