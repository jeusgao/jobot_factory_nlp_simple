#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 16:48:17
# @Author  : Joe Gao (jeusgao@163.com)

import numpy as np
from backend import keras, K, tf
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
        min_lr=min_lr,
        learning_rate=lr,
    )


def export_to_custom_objects(base_extend_with):
    """装饰器，用来将优化器放到custom_objects中
    """
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if isinstance(name, str):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        keras.utils.get_custom_objects()[name] = NewOptimizer

        return NewOptimizer

    return new_extend_with


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


@export_to_custom_objects
def extend_with_exponential_moving_average(BaseOptimizer):
    """返回新的优化器类，加入EMA（权重滑动平均）
    """
    class NewOptimizer(BaseOptimizer):
        """带EMA（权重滑动平均）的优化器
        """
        @insert_arguments(ema_momentum=0.999)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def get_updates(self, loss, params):
            updates = super(NewOptimizer, self).get_updates(loss, params)
            self.model_weights = params
            self.ema_weights = [K.zeros(K.shape(w)) for w in params]
            self.old_weights = K.batch_get_value(params)

            ema_updates, ema_momentum = [], self.ema_momentum
            with tf.control_dependencies(updates):
                for w1, w2 in zip(self.ema_weights, params):
                    new_w = ema_momentum * w1 + (1 - ema_momentum) * w2
                    ema_updates.append(K.update(w1, new_w))

            return ema_updates

        def get_config(self):
            config = {
                'ema_momentum': self.ema_momentum,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def apply_ema_weights(self, bias_correction=True):
            """备份原模型权重，然后将平均权重应用到模型上去。
            """
            self.old_weights = K.batch_get_value(self.model_weights)
            ema_weights = K.batch_get_value(self.ema_weights)

            if bias_correction:
                iterations = K.eval(self.iterations)
                scale = 1.0 - np.power(self.ema_momentum, iterations)
                ema_weights = [weight / scale for weight in ema_weights]

            K.batch_set_value(zip(self.model_weights, ema_weights))

        def reset_old_weights(self):
            """恢复模型到旧权重。
            """
            K.batch_set_value(zip(self.model_weights, self.old_weights))

    return NewOptimizer
