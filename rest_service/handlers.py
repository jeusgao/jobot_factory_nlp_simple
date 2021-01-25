#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
from fastapi import FastAPI

from predictor import main

app = FastAPI()


@app.get("/{api_name}")
async def pred(api_name: str, input1: str, input2: str=None):

    rst = main(api_name, input1, input2)

    return rst
