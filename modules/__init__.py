#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-07 13:10:57
# @Author  : Joe Gao (jeusgao@163.com)

from backend import keras, V_TF
from keras_bert.layers import MaskedGlobalMaxPool1D

from .models import base_embed, bert_base, get_model, nonmask_embed
from .optimizers import adam, adam_warmup, AdamEMA
from .tokenizers import tokenizer_zh, kwr_labeler, cls_labeler
from .callbacks import TrainingCallbacks, EvaluatingCallbacks
from .generators import spo_data_generator_train, data_generator_train, data_generator_pred
from .funcs import gather_words, get_square, init_truncated_normal
from .layers import (
    base_inputs,
    NonMaskingLayer,
    bi_gru,
    dropout,
    layer_normalization,
    batch_normalization,
    reshape,
)
from keras_metrics import (
    categorical_precision,
    categorical_recall,
    categorical_f1_score,
    binary_precision,
    binary_recall,
    binary_f1_score,
)

DIC_Inits = {
    'truncated': {'func': init_truncated_normal}
}

DIC_Funcs = {
    'gather_words': gather_words,
    'get_square': get_square,
}

DIC_Losses = {
    'categorical_crossentropy': {
        'func': keras.losses.CategoricalCrossentropy(),
        # 'params': {'from_logits': False},
    },
    'binarycrossentropy': {
        'func': keras.losses.BinaryCrossentropy(),
        # 'params': {'from_logits': False},
    },
    'sparse_categorical_crossentropy': {
        'func': keras.losses.SparseCategoricalCrossentropy(),
    }
}

DIC_Metrics = {
    # 'categorical_precision': {'func': categorical_precision, 'params': {'label': 0}},
    # 'categorical_recall': {'func': categorical_recall, 'params': {'label': 0}},
    # 'categorical_f1_score': {'func': categorical_f1_score, 'params': {'label': 0}},
    'accuracy': {'func': 'accuracy'},
    'binary_accuracy': {'func': keras.metrics.BinaryAccuracy()},
    'binary_precision': {'func': binary_precision()},
    'binary_recall': {'func': binary_recall()},
    'binary_f1_score': {'func': binary_f1_score()},
    'categorical_accuracy': {'func': keras.metrics.CategoricalAccuracy()},
    'precision': {'func': keras.metrics.Precision()},
    'recall': {'func': keras.metrics.Recall()},
    'sparse_categorical_accuracy': {'func': keras.metrics.SparseCategoricalAccuracy()},
}

DIC_Layers = {
    # 'base_inputs': {'func': base_inputs},
    'nonmasking_layer': {'func': NonMaskingLayer},
    'input': {'func': keras.layers.Input, 'params': {'shape': (None,)}},
    'dense': {'func': keras.layers.Dense, 'params': {'units': 64, 'activation': 'relu'}},
    'lambda': {'func': keras.layers.Lambda, 'params': ''},
    'bigru': {'func': bi_gru, 'params': {'units': 64, 'return_sequences': True, 'reset_after': True}},
    'dropout': {'func': dropout, 'params': {'rate': 0.1}},
    'masked_global_max_pool1D': {'func': MaskedGlobalMaxPool1D, 'params': {'name': 'Masked-Global-Pool-Max'}},
    'layer_normalization': {'func': layer_normalization, 'params': {'axis': -1, 'epsilon': 0.001, }},
    'batch_normalization': {'func': batch_normalization, 'params': {'axis': -1, 'epsilon': 0.001, }},
    'reshape': {'func': reshape, 'params': [1, 1]}
}

if V_TF >= '2.2':
    from .layers import KConditionalRandomField
    kcrf = KConditionalRandomField()
    DIC_Layers['kcrf'] = {'func': kcrf}
    DIC_Losses['kcrf_loss'] = {'func': kcrf.loss}
    DIC_Metrics['kcrf_accuracy'] = {'func': kcrf.accuracy}
else:
    from keras_contrib.losses import crf_loss
    from keras_contrib.metrics import crf_viterbi_accuracy as crf_accuracy
    from .layers import crf
    DIC_Layers['crf'] = {'func': crf, 'params': {'dim': 2}}
    DIC_Losses['crf_loss'] = {'func': crf_loss}
    DIC_Metrics['crf_accuracy'] = {'func': crf_accuracy}

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
    'base_embed': {'func': base_embed},
    'nonmask_embed': {'func': nonmask_embed},
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
    'AdamEMA': {
        'func': AdamEMA,
        'params': {
            'learning_rate': 1e-6,
        }
    },
}

DIC_Tokenizers = {
    'tokenizer_zh': {'func': tokenizer_zh, 'params': {'fn_vocab': None}},
}

DIC_Labelers = {
    'kwr_labeler': {'func': kwr_labeler},
    'cls_labeler': {'func': cls_labeler},
}

DIC_Generators_for_train = {
    'spo_data_generator_train': {
        'func': spo_data_generator_train,
        'params': {
            'data': None,
            'Y': None,
            'tokenizer': None,
            'dim': 2,
            'maxlen': 512,
            'labeler': None,
            'activation': 'sigmoid',
        }
    },
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
