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


class Predictor(object):
    def __init__(self, api_name):
        task_path = f'hub/models/{api_name}'
        fn_model, params_model, params_data, _, params_pred = task_init(task_path, is_train=False)

        if params_model.get('TF_KERAS', 0) == 1:
            os.environ["TF_KERAS"] = '1'

        self.labeler = None
        self.maxlen = params_model.get('maxlen')
        self.ML = params_model.get('ML')
        self.is_pair = False if self.ML == self.maxlen else True
        self.activation = params_data.get('activation')
        self.is_sequence = params_data.get('is_sequence')

        fn_labeler = params_data.get('fn_labeler')
        if fn_labeler:
            fn_labeler = f'{task_path}/{fn_labeler}'
            if os.path.exists(fn_labeler):
                self.labeler = pickle.load(open(fn_labeler, 'rb'))

        self.data_generator = DIC_Generators_for_pred.get(params_data.get('data_generator_for_pred')).get('func')
        self.resolver = DIC_Resolvers.get(params_pred.get('resolver')).get('func')

        self.tokenizer, self.token_dict, self.model = model_builder(is_eval=True, **params_model)
        # self.model.summary()
        self.model.load_weights(fn_model)

    def predict(self, inputs):
        print(inputs)
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


v_lower = ['ner']
v_higher = ['ner_23']
include = v_lower if V_TF < 2.2 else v_higher
models = tuple(os.walk('hub/models'))[0][1]

DIC_Predictors = {
    k: Predictor(k) for k in models if os.path.exists(f'hub/models/{k}/model.h5') and k in include
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
