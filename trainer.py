#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

import os
import json
import time
import pickle
import argparse

import tensorflow as tf
from keras_bert import calc_train_steps

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from contextlib import redirect_stdout

from builders import model_builder, train_data_builder
from utils import (
    TrainingCallbacks,
    EvaluatingCallbacks,
    DIC_Generators,
    data_generator_eval,
    trainer_init,
)


def main(
    fn_model,
    target_path=None,
    params_model=None,
    params_data=None,
    params_train=None,
    action='training',
):
    maxlen = params_model.get('maxlen')
    ML = params_model.get('ML')

    batch_size = params_data.get('batch_size')
    activation = params_data.get('activation')
    fn_labeler = params_data.get('fn_labeler')

    is_eval = action == 'evaluating'

    fn_weights = fn_model if is_eval else params_train.get('checkpoint')

    labeler = None

    if fn_labeler:
        fn_labeler = f'{target_path}/{fn_labeler}'
        labeler = {}
        if os.path.exists(fn_labeler):
            labeler = pickle.load(open(fn_labeler, 'rb'))
        dic_labeler = DIC_Tokenizers.get('kwr_labeler')
        labeler = dic_labeler.get('func')(labeler=labeler, y_data=train_y + valid_y)

    tokenizer, model = model_builder(**params_model)

    if fn_weights:
        model.load_weights(f'{fn_weights}')

    with open(f'{target_path}/model_structs.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    data_generator = DIC_Generators.get(params_data.get('data_generator')).get('func')
    dim = len(labeler) if labeler else 2

    if is_eval:
        evaluate_callacks = EvaluatingCallbacks(task_path=target_path, log_name=action)
        test_x, test_y, test_steps = train_data_builder(
            data_loader_params=params_data.get('data_loader_params'),
            fns=params_data.get('fns_test'),
            batch_size=batch_size,
        )

        test_D = data_generator(
            data=test_x,
            y_data=test_y,
            tokenizer=tokenizer,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=128,
            labeler=labeler,
            activation=activation,
        )

        hist = model.evaluate(
            test_D,
            verbose=1,
            steps=test_steps,
            callbacks=[
                evaluate_callacks,
            ],
        )
        return hist

    else:

        train_x, train_y, train_steps = train_data_builder(
            data_loader_params=params_data.get('data_loader_params'),
            fns=params_data.get('fns_train'),
            batch_size=batch_size,
        )
        valid_x, valid_y, valid_steps = train_data_builder(
            data_loader_params=params_data.get('data_loader_params'),
            fns=params_data.get('fns_dev'),
            batch_size=batch_size,
        )
        total_steps, warmup_steps = calc_train_steps(
            num_example=len(train_x),
            batch_size=batch_size,
            epochs=5,
            warmup_proportion=0.1,
        )

        train_D = data_generator(
            data=train_x,
            y_data=train_y,
            tokenizer=tokenizer,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=batch_size,
            labeler=labeler,
            activation=activation,
        )

        valid_D = data_generator(
            data=valid_x,
            y_data=valid_y,
            tokenizer=tokenizer,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=batch_size,
            labeler=labeler,
            activation=activation,
        )

        cp_loss = keras.callbacks.ModelCheckpoint(fn_model, **params_train.get('cp_loss'))
        early_stopping = keras.callbacks.EarlyStopping(**params_train.get('early_stopping'))
        train_callbacks = TrainingCallbacks(task_path=target_path)

        hist = model.fit(
            train_D,
            steps_per_epoch=train_steps,
            epochs=params_train.get('epochs'),
            validation_data=valid_D,
            validation_steps=valid_steps,
            verbose=1,
            callbacks=[
                cp_loss,
                early_stopping,
                train_callbacks,
            ],
        )
        return hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args of trainer")
    parser.add_argument("-p", "--path", default="")
    parser.add_argument("-a", "--action", default="training")
    args = parser.parse_args()

    target_path = args.path
    action = args.action

    fn_model, params_model, params_data, params_train = trainer_init(target_path)

    main(
        fn_model,
        target_path=target_path,
        params_model=params_model,
        params_data=params_data,
        params_train=params_train,
        action=action,
    )
