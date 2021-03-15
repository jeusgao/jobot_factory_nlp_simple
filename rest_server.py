#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe Gao (jeusgao@163.com)

import os
import uvicorn
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args of rest service")
    parser.add_argument("--host", default='0.0.0.0')
    parser.add_argument("-p", "--port", type=int, default=9055)
    parser.add_argument("-r", "--reload", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run('rest_service.handlers:app', host=args.host, port=args.port, reload=args.reload == 1)
