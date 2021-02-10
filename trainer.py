#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import json
import time
import pickle
import argparse
from sklearn.utils import shuffle

from backend import keras
from contextlib import redirect_stdout
from keras_bert import calc_train_steps

from builders import model_builder, train_data_builder
from utils import task_init
from modules import (
    DIC_Labelers,
    TrainingCallbacks,
    EvaluatingCallbacks,
    DIC_Generators_for_train,
)


def main(
    fn_model,
    target_path=None,
    params_model=None,
    params_data=None,
    params_train=None,
    action='training',
):
    if params_model.get('TF_KERAS', 0) == 1:
        os.environ["TF_KERAS"] = '1'

    maxlen = params_model.get('maxlen')
    ML = params_model.get('ML')

    batch_size = params_data.get('batch_size')
    activation = params_data.get('activation')
    fn_labeler = params_data.get('fn_labeler')
    is_sequence = params_data.get('is_sequence')

    is_eval = action == 'evaluating'

    fn_weights = fn_model if is_eval else params_train.get('checkpoint')

    labeler = None
    if fn_labeler:
        fn_labeler = f'{target_path}/{fn_labeler}'
        if os.path.exists(fn_labeler):
            labeler = pickle.load(open(fn_labeler, 'rb'))
        else:
            labeler = {
                'O': 0,
            }

    tokenizer, token_dict, model = model_builder(is_eval=is_eval, **params_model)

    if fn_weights:
        model.load_weights(f'{fn_weights}')

    with open(f'{target_path}/model_structs.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    data_generator = DIC_Generators_for_train.get(params_data.get('data_generator_for_train')).get('func')
    dim = len(labeler) if labeler else 2

    test_x, test_y, test_steps = train_data_builder(
        data_loader_params=params_data.get('data_loader_params'),
        fns=params_data.get('fns_test'),
        batch_size=batch_size,
    )
    test_x, test_y = shuffle(test_x, test_y, random_state=0)

    if is_eval:
        evaluate_callacks = EvaluatingCallbacks(task_path=target_path, log_name=action)

        test_D = data_generator(
            data=test_x,
            y_data=test_y,
            tokenizer=tokenizer,
            token_dict=token_dict,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=128,
            labeler=labeler,
            activation=activation,
            is_sequence=is_sequence,
        )

        model.evaluate(
            test_D,
            verbose=1,
            steps=test_steps,
            callbacks=[
                evaluate_callacks,
            ],
        )

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
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        valid_x, valid_y = shuffle(valid_x, valid_y, random_state=0)
        print(len(train_x), len(valid_x))

        if fn_labeler and is_sequence:
            func = DIC_Labelers.get('kwr_labeler').get('func')
            labeler = func(labeler=labeler, y_data=train_y + valid_y + test_y)
            dim = len(labeler)
            pickle.dump(labeler, open(fn_labeler, 'wb'))
            print(labeler)

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
            token_dict=token_dict,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=batch_size,
            labeler=labeler,
            activation=activation,
            is_sequence=is_sequence,
        )

        valid_D = data_generator(
            data=valid_x,
            y_data=valid_y,
            tokenizer=tokenizer,
            token_dict=token_dict,
            dim=dim,
            maxlen=maxlen,
            ML=ML,
            batch_size=batch_size,
            labeler=labeler,
            activation=activation,
            is_sequence=is_sequence,
        )

        cp_loss = keras.callbacks.ModelCheckpoint(fn_model, **params_train.get('cp_loss'))
        early_stopping = keras.callbacks.EarlyStopping(**params_train.get('early_stopping'))
        train_callbacks = TrainingCallbacks(task_path=target_path)

        model.fit(
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args of trainer")
    parser.add_argument("-p", "--path", default="")
    parser.add_argument("-a", "--action", default="training")
    args = parser.parse_args()

    target_path = args.path
    action = args.action

    fn_model, params_model, params_data, params_train, _ = task_init(target_path)

    main(
        fn_model,
        target_path=target_path,
        params_model=params_model,
        params_data=params_data,
        params_train=params_train,
        action=action,
    )
