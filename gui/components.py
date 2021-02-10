#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-15 09:43:28
# @Author  : Joe Gao (jeusgao@163.com)

import os
import streamlit as st


def get_default_input(_default, params, tag, sub_tag=None):
    if params:
        _tmp = params.get(tag).get(sub_tag) if sub_tag else params.get(tag)
        if _tmp:
            _default = _tmp
    return _default


def get_default(params, dic, tag, is_num=False):
    _options = list(dic.keys())
    _default = 0 if is_num else _options[0]
    if params:
        if is_num:
            _tmp = params.get(tag)
            if _tmp:
                _default = _options.index(_tmp)
        else:
            _default = params.get(tag)
    return _options, _default


def get_default_params(params, dic, tag, sub_tag=None):
    _options = list(dic.keys())
    _default = _options[0]
    _params = None
    if params and params.get(tag):
        if sub_tag:
            if params.get(tag).get(sub_tag):
                _default = params.get(tag).get(sub_tag).get('func')
                _params = params.get(tag).get(sub_tag).get('params')
        else:
            _default = params.get(tag).get('func')
            _params = params.get(tag).get('params')

    return _options, _default, _params


def single_option(tag, dic, main_dic, _options, _params=None, _index=0):
    _option = st.selectbox('', _options, index=_index)
    if not _params:
        _params = dic.get(_option).get('params')
    if _params:
        _params = eval(st.text_input(f'{_option} params:', _params))
        dic[_option]['params'] = _params
        main_dic[tag] = {'func': _option, 'params': _params}
    else:
        st.write(f'{_option}: None params')
        main_dic[tag] = {'func': _option}
    return main_dic


def multi_options(task_path, tag, options, main_dic, _default=None, fn=None, is_set=False, is_params=True, tpl_dic=None):
    _keys = []
    if is_set:
        _keys = st.multiselect(tag, options, default=_default if _default in options else options[0], key=tag)
    else:
        fn = f'{task_path}/{fn}'
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                _keys = f.read().splitlines()

        for key in options:
            if st.button(key, key=key):
                _keys.append(key)
        _keys = st.text_area(f'{tag} structure:', '\n'.join(_keys), height=200).splitlines()

        with open(fn, 'w') as f:
            f.write('\n'.join(filter(None, _keys)))

    if is_params:
        _ori_funcs = main_dic.get(tag)
        tmps = []
        for i, key in enumerate(_keys):
            if not key:
                continue

            _params = tpl_dic.get(key) if tpl_dic else None

            if _ori_funcs and len(_ori_funcs):
                if i < len(_ori_funcs) and _ori_funcs[i].get('func'):
                    if _ori_funcs[i].get('func') == key:
                        _params = _ori_funcs[i]

            if _params or _params.get('params'):
                _params = eval(st.text_input(f'{key} params:', _params.get('params'), key=f'{tag}_{key}_params_{i}'))
            else:
                st.write(f'{key}: None param')
                _params = None

            if _params:
                tmp = {'func': key, 'params': _params}
            else:
                tmp = {'func': key}

            tmps.append(tmp)
    else:
        tmps = _keys

    main_dic[tag] = tmps

    return main_dic
