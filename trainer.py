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
    action='training',
    model_bases_params=None,
    model_common_params=None,
    model_embeded_params=None,
    model_inputs_params=None,
    model_layer_params=None,
    model_outputs_params=None,
    model_optimizer_params=None,
    params_data=None,
    params_train=None,
    params_pred=None,
):
    if model_common_params.TF_KERAS == 1:
        os.environ["TF_KERAS"] = '1'

    maxlen = model_common_params.maxlen
    ML = model_common_params.ML

    batch_size = params_data.batch_size
    activation = params_data.activation
    fn_labeler = params_data.fn_labeler
    is_sequence = params_data.is_sequence

    is_eval = action == 'evaluating'

    fn_weights = fn_model if is_eval else params_train.checkpoint
    # fn_weights = fn_model

    labeler = None
    if fn_labeler:
        fn_labeler = f'{target_path}/{fn_labeler}'
        if os.path.exists(fn_labeler):
            labeler = pickle.load(open(fn_labeler, 'rb'))
        else:
            labeler = {
                'O': 0,
            }
###
    params_model = {}
    params_model['is_eval'] = is_eval
    params_model['maxlen'] = maxlen
    params_model['ML'] = ML
    params_model['tokenizer_code'] = 'tokenizer_zh'
    params_model['tokenizer_params'] = {'fn_vocab': 'hub/bases/rbtl3/vocab.txt'}
    params_model['obj_common'] = model_common_params
    params_model['dic_bases'] = model_bases_params
    params_model['dic_embeds'] = model_embeded_params
    params_model['list_inputs'] = model_inputs_params
    params_model['dic_layers'] = model_layer_params
    params_model['dic_outputs'] = model_outputs_params
    params_model['obj_optimizer'] = model_optimizer_params

    tokenizer, token_dict, model = model_builder(**params_model)

    if fn_weights:
        model.load_weights(f'{fn_weights}')
        print('Weights loaded.')

    with open(f'{target_path}/model_structs.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
        print('model structs saved.')

    data_generator = DIC_Generators_for_train.get(params_data.data_generator_for_train).get('func')
    dim = len(labeler) if labeler else 2

    test_x, test_y, test_steps = train_data_builder(
        data_loader_params=params_data.data_loader_params,
        fns=params_data.fns_test,
        batch_size=batch_size,
    )
    test_x, test_y = shuffle(test_x, test_y, random_state=0)
    print(f'Test data lenth: {len(test_x)}.')

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
            data_loader_params=params_data.data_loader_params,
            fns=params_data.fns_train,
            batch_size=batch_size,
        )
        valid_x, valid_y, valid_steps = train_data_builder(
            data_loader_params=params_data.data_loader_params,
            fns=params_data.fns_dev,
            batch_size=batch_size,
        )
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        valid_x, valid_y = shuffle(valid_x, valid_y, random_state=0)
        print(len(train_x), len(valid_x))

        if fn_labeler:
            func = DIC_Labelers.get('kwr_labeler').get('func') if is_sequence else DIC_Labelers.get('cls_labeler').get('func')
            labeler = func(labeler=labeler, y_data=train_y + valid_y + test_y)
            dim = len(labeler)
            pickle.dump(labeler, open(fn_labeler, 'wb'))

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

        cp_loss = keras.callbacks.ModelCheckpoint(fn_model, **params_train.cp_loss)
        early_stopping = keras.callbacks.EarlyStopping(**params_train.early_stopping)
        train_callbacks = TrainingCallbacks(task_path=target_path)

        model.fit(
            train_D,
            steps_per_epoch=train_steps,
            epochs=params_train.epochs,
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

    fn_model, dic_task_params = task_init(target_path)

    main(
        fn_model,
        target_path=target_path,
        action=action,
        **dic_task_params,
    )
