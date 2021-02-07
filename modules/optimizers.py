#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 16:48:17
# @Author  : Joe Gao (jeusgao@163.com)

from backend import keras
from keras_bert import AdamWarmup, calc_train_steps


def adam(lr=1e-4):
    return keras.optimizers.Adam(lr)


def adam_warmup(len_data=1000, batch_size=128, epochs=5, warmup_proportion=0.1, lr=1e-4, min_lr=1e-5):
    total_steps, warmup_steps = calc_train_steps(
        num_example=len_data,
        batch_size=batch_size,
        epochs=epochs,
        warmup_proportion=warmup_proportion,
    )
    return AdamWarmup(
        total_steps,
        warmup_steps,
        lr=lr,
        min_lr=min_lr,
    )
