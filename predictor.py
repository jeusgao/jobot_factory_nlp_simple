#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-23 09:15:41
# @Author  : Joe Gao (jeusgao@163.com)
# @Link    : https://www.jianshu.com/u/3b77f85cc918
# @Version : $Id$

import os
from builders import model_builder
from utils import DIC_Generators_for_pred, DIC_Resolvers, task_init


class Predictor(object):
    def __init__(self, api_name):
        task_path = f'hub/models/{api_name}'
        fn_model, params_model, params_data, _, params_pred = task_init(task_path, is_train=False)

        self.labeler, self.tokenizer, self.model = None, None, None
        self.maxlen = params_model.get('maxlen')
        self.ML = params_model.get('ML')

        self.is_sequence = params_data.get('is_sequence')

        fn_labeler = params_data.get('fn_labeler')
        if fn_labeler:
            fn_labeler = f'{task_path}/{fn_labeler}'
            if os.path.exists(fn_labeler):
                self.labeler = pickle.load(open(fn_labeler, 'rb'))

        self.data_generator = DIC_Generators_for_pred.get(params_data.get('data_generator_for_pred')).get('func')
        self.resolver = DIC_Resolvers.get(params_pred.get('resolver')).get('func')

        self.tokenizer, self.model = model_builder(**params_model)
        # self.model.summary()
        self.model.load_weights(fn_model)

    def predict(self, inputs):
        # data = inputs if self.is_sequence else [inputs]
        data_input = self.data_generator(
            data=inputs,
            tokenizer=self.tokenizer,
            maxlen=self.maxlen,
            ML=self.ML,
        )
        pred = self.model.predict(data_input)
        rst = self.resolver(pred, inputs, labeler=self.labeler, is_sequence=self.is_sequence)
        return rst


models = tuple(os.walk('hub/models'))[0][1]

DIC_Predictors = {
    k: Predictor(k) for k in models if os.path.exists(f'hub/models/{k}/model.h5')
}


def main(api_name: str, input1: str, input2: str=None):
    inputs = [input1]
    if input2:
        inputs.append(input2)
    predictor = DIC_Predictors.get(api_name)
    rst = predictor.predict(inputs)
    print('+++++', rst)

    return rst
