#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
from fastapi import FastAPI

from predictor import main

app = FastAPI()


@app.get("/pred/{api_name}")
async def pred(api_name, inputs):

    rst = main(api_name, inputs)

    return {'result': rst}
