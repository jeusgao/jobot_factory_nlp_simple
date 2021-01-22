#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
from fastapi import FastAPI

app = FastAPI()


@app.post("/qqp_predict/")
async def qqp_pred(text1: str, text1: str):
    rst = None
    return {'result': rst}
