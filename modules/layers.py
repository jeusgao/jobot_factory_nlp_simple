#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-04 16:46:37
# @Author  : Joe Gao (jeusgao@163.com)

from distutils.version import LooseVersion
from backend import keras, V_TF, K
from keras_contrib.layers import CRF
from keras_bert.layers import MaskedGlobalMaxPool1D, EmbeddingSimilarity, Masked

if V_TF >= '2.3':
    import tensorflow as tf
    import tensorflow_addons as tfa

    class KConditionalRandomField(keras.layers.Layer):
        """
        K is to mark Kashgari version of CRF
        Conditional Random Field layer (tf.keras)
        `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
        must be equal to the number of classes the CRF can predict (a linear layer is recommended).

        Args:
            num_labels (int): the number of labels to tag each temporal input.

        Input shape:
            nD tensor with shape `(batch_size, sentence length, num_classes)`.

        Output shape:
            nD tensor with shape: `(batch_size, sentence length, num_classes)`.

        Masking
            This layer supports keras masking for input data with a variable number
            of timesteps. To introduce masks to your data,
            use an embedding layer with the `mask_zero` parameter
            set to `True` or add a Masking Layer before this Layer
        """

        def __init__(self,
                     sparse_target=True,
                     **kwargs):
            if LooseVersion(V_TF) < '2.2.0':
                raise ImportError("The KConditionalRandomField requires TensorFlow 2.2.x version or higher.")

            super().__init__()
            self.transitions = kwargs.pop('transitions', None)
            self.output_dim = kwargs.pop('output_dim', None)
            self.sparse_target = sparse_target
            self.sequence_lengths = None
            self.mask = None

        def get_config(self):
            config = {
                "output_dim": self.output_dim,
                "transitions": K.eval(self.transitions),
            }
            base_config = super().get_config()
            return dict(**base_config, **config)

        def build(self, input_shape):
            self.output_dim = input_shape[-1]
            assert len(input_shape) == 3
            self.transitions = self.add_weight(
                name="transitions",
                shape=[input_shape[-1], input_shape[-1]],
                initializer="glorot_uniform",
                trainable=True
            )

        def call(self, inputs, mask=None, **kwargs):
            if mask is not None:
                self.sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
                self.mask = mask
            else:
                self.sequence_lengths = K.sum(K.ones_like(inputs[:, :, 0], dtype='int32'), axis=-1)
            viterbi_sequence, _ = tfa.text.crf_decode(
                inputs, self.transitions, self.sequence_lengths
            )
            output = K.cast(K.one_hot(viterbi_sequence, inputs.shape[-1]), inputs.dtype)
            return K.in_train_phase(inputs, output)

        def loss(self, y_true, y_pred):
            if len(K.int_shape(y_true)) == 3:
                y_true = K.argmax(y_true, axis=-1)
            log_likelihood, self.transitions = tfa.text.crf_log_likelihood(
                y_pred,
                y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return tf.reduce_mean(-log_likelihood)

        def compute_output_shape(self, input_shape):
            return input_shape[:2] + (self.out_dim,)

        # use crf decode to estimate accuracy
        def accuracy(self, y_true, y_pred):
            mask = self.mask
            if len(K.int_shape(y_true)) == 3:
                y_true = K.argmax(y_true, axis=-1)

            y_pred, _ = tfa.text.crf_decode(
                y_pred, self.transitions, self.sequence_lengths
            )
            y_true = K.cast(y_true, y_pred.dtype)
            is_equal = K.equal(y_true, y_pred)
            is_equal = K.cast(is_equal, y_pred.dtype)
            if mask is None:
                return K.mean(is_equal)
            else:
                mask = K.cast(mask, y_pred.dtype)
                return K.sum(is_equal * mask) / K.sum(mask)

        # Use argmax to estimate accuracy
        def fast_accuracy(self, y_true, y_pred):
            mask = self.mask
            if len(K.int_shape(y_true)) == 3:
                y_true = K.argmax(y_true, axis=-1)
            y_pred = K.argmax(y_pred, -1)
            y_true = K.cast(y_true, y_pred.dtype)
            # 逐标签取最大来粗略评测训练效果
            isequal = K.equal(y_true, y_pred)
            isequal = K.cast(isequal, y_pred.dtype)
            if mask is None:
                return K.mean(isequal)
            else:
                mask = K.cast(mask, y_pred.dtype)
                return K.sum(isequal * mask) / K.sum(mask)
else:
    def crf(dim=2):
        return CRF(dim)


class NonMaskingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, input_mask=None):
        return None

    def call(self, x, mask=None):
        return x


class LayerNormalization(keras.layers.Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """

    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.hidden_initializer = keras.initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    # @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'hidden_initializer':
                keras.initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def bi_gru(**params):
    return keras.layers.Bidirectional(keras.layers.GRU(**params))


def dropout(rate=0.1):
    return keras.layers.Dropout(rate)


def base_inputs(base):
    return base.inputs, base.output


def layer_normalization(**params):
    return LayerNormalization(**params)


def batch_normalization(**params):
    return keras.layers.BatchNormalization(**params)


def reshape(param):
    param = tuple(param)
    return keras.layers.Reshape(param)
