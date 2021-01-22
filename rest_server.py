#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
import uvicorn


if __name__ == '__main__':

    uvicorn.run('rest_service.handlers:app', port=9060, reload=True)
