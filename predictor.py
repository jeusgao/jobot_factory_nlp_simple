#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-23 09:15:41
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
import pickle
from builders import model_builder
from modules import DIC_Generators_for_pred
from utils import DIC_Resolvers, task_init
from backend import V_TF


def _get_params(
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
    return model_bases_params, model_common_params, model_embeded_params, model_inputs_params, model_layer_params, model_outputs_params, model_optimizer_params, params_data, params_train, params_pred


class Predictor(object):
    def __init__(self, api_name):
        task_path = f'hub/models/{api_name}'
        fn_model, dic_task_params = task_init(task_path, is_train=False)

        model_bases_params, model_common_params, model_embeded_params, model_inputs_params, model_layer_params, model_outputs_params, model_optimizer_params, params_data, params_train, params_pred = _get_params(
            **dic_task_params)

        if model_common_params.TF_KERAS == 1:
            os.environ["TF_KERAS"] = '1'

        self.labeler = None
        self.maxlen = model_common_params.maxlen
        self.ML = model_common_params.ML
        self.is_pair = model_common_params.is_pair
        self.activation = params_data.activation
        self.is_sequence = params_data.is_sequence

        fn_labeler = params_data.fn_labeler
        if fn_labeler:
            fn_labeler = f'{task_path}/{fn_labeler}'
            if os.path.exists(fn_labeler):
                self.labeler = pickle.load(open(fn_labeler, 'rb'))

        self.data_generator = DIC_Generators_for_pred.get(params_data.data_generator_for_pred).get('func')
        self.resolver = DIC_Resolvers.get(params_pred.resolver).get('func')

        params_model = {}
        params_model['maxlen'] = self.maxlen
        params_model['ML'] = self.ML
        params_model['tokenizer_code'] = 'tokenizer_zh'
        params_model['tokenizer_params'] = {'fn_vocab': 'hub/bases/rbtl3/vocab.txt'}
        params_model['obj_common'] = model_common_params
        params_model['dic_bases'] = model_bases_params
        params_model['dic_embeds'] = model_embeded_params
        params_model['list_inputs'] = model_inputs_params
        params_model['dic_layers'] = model_layer_params
        params_model['dic_outputs'] = model_outputs_params
        params_model['obj_optimizer'] = model_optimizer_params

        self.tokenizer, self.token_dict, self.model = model_builder(is_predict=True, **params_model)
        # self.model.summary()
        self.model.load_weights(fn_model)

    def predict(self, inputs):
        if self.is_pair:
            if len(inputs) < 2:
                return {'result': 'Not enough inputs.'}
        elif len(inputs) > 1:
            inputs = ['.'.join(inputs)]

        data_input = self.data_generator(
            data=inputs,
            tokenizer=self.tokenizer,
            token_dict=self.token_dict,
            maxlen=self.maxlen,
            ML=self.ML,
            is_sequence=self.is_sequence,
        )
        pred = self.model.predict(data_input)
        rst = self.resolver(pred, inputs, activation=self.activation, labeler=self.labeler, is_sequence=self.is_sequence)
        return rst


models = tuple(os.walk('hub/models'))[0][1]

DIC_Predictors = {
    k: Predictor(k) for k in models if os.path.exists(f'hub/models/{k}/model.h5')
}


def main(api_name, input1, input2=None):
    if not len(input1.strip()):
        return {'result': 'Empty input(s).'}

    inputs = [input1]
    if input2 and len(input2.strip()):
        inputs.append(input2)
    predictor = DIC_Predictors.get(api_name)
    rst = predictor.predict(inputs)

    return rst
