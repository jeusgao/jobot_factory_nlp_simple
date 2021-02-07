#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

from backend import keras
from keras_bert.layers import MaskedGlobalMaxPool1D

from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy as crf_accuracy

from .models import base_embed, bert_base, get_model
from .optimizers import adam, adam_warmup
from .tokenizers import tokenizer_zh, kwr_labeler
from .callbacks import TrainingCallbacks, EvaluatingCallbacks
from .generators import data_generator_train, data_generator_pred

from .layers import (
    KConditionalRandomField,
    crf,
    base_inputs,
    # nonmasking_layer,
    NonMaskingLayer,
    bi_gru,
    dropout,
)
from keras_metrics import (
    categorical_precision,
    categorical_recall,
    categorical_f1_score,
    binary_precision,
    binary_recall,
    binary_f1_score,
)

kcrf = KConditionalRandomField()

DIC_Losses = {
    'kcrf_loss': {'func': kcrf.loss},
    'crf_loss': {'func': crf_loss},
    'categorical_crossentropy': {
        'func': keras.losses.CategoricalCrossentropy,
        'params': {'from_logits': False},
    },
    'binarycrossentropy': {
        'func': keras.losses.BinaryCrossentropy,
        'params': {'from_logits': False},
    },
}

DIC_Metrics = {
    'categorical_precision': {'func': categorical_precision, 'params': {'label': 0}},
    'categorical_recall': {'func': categorical_recall, 'params': {'label': 0}},
    'categorical_f1_score': {'func': categorical_f1_score, 'params': {'label': 0}},
    'binary_precision': {'func': binary_precision()},
    'binary_recall': {'func': binary_recall()},
    'binary_f1_score': {'func': binary_f1_score()},
    'kcrf_accuracy': {'func': kcrf.accuracy},
    'crf_accuracy': {'func': crf_accuracy},
    'accuracy': {'func': 'accuracy'},
}

DIC_Layers = {
    'base_inputs': {'func': base_inputs},
    'nonmasking_layer': {'func': NonMaskingLayer()},
    'input': {'func': keras.layers.Input, 'params': {'shape': (None,)}},
    'dense': {'func': keras.layers.Dense, 'params': {'units': 64, 'activation': 'relu'}},
    'bigru': {'func': bi_gru, 'params': {'units': 64, 'return_sequences': True, 'reset_after': True}},
    'dropout': {'func': dropout, 'params': {'rate': 0.1}},
    'crf': {'func': crf, 'params': {'dim': 2}},
    'kcrf': {'func': kcrf},
    'masked_global_max_pool1D': {'func': MaskedGlobalMaxPool1D, 'params': {'name': 'Masked-Global-Pool-Max'}},
}

DIC_Bases = {
    'BERT': {
        'func': bert_base,
        'params': {
            'fn_config': None,
            'fn_base_model': None,
            'training': False,
            'trainable': False,
            'seq_len': 512
        }
    },
}

DIC_Models = {
    'base_embed': base_embed,
}

DIC_Optimizers = {
    'adam': {'func': adam, 'params': {'lr': 1e-4}},
    'adam_warmup': {
        'func': adam_warmup,
        'params': {
            'len_data': 1000,
            'batch_size': 128,
            'epochs': 5,
            'warmup_proportion': 0.1,
            'lr': 1e-4,
            'min_lr': 1e-5,
        }
    },
}

DIC_Tokenizers = {
    'tokenizer_zh': {'func': tokenizer_zh, 'params': {'fn_vocab': None}},
}

DIC_Labelers = {
    'kwr_labeler': {'func': kwr_labeler},
}

DIC_Generators_for_train = {
    'data_generator_for_train': {
        'func': data_generator_train,
        'params': {
            'data': None,
            'Y': None,
            'tokenizer': None,
            'dim': 2,
            'maxlen': 512,
            'labeler': None,
            'activation': 'sigmoid',
        },
    }
}

DIC_Generators_for_pred = {
    'data_generator_for_pred': {
        'func': data_generator_pred,
    }
}
