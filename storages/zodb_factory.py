#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-25 16:10:18
# @Author  : Joe Gao (jeusgao@163.com)

import os

import ZODB
import ZODB.FileStorage as zfs
from BTrees import IOBTree
import zc.zlibstorage
import transaction


class ZodbFactory(object):
    def __init__(self, db_path, compress=True):
        storage = zfs.FileStorage(db_path)
        compressed_storage = zc.zlibstorage.ZlibStorage(storage, compress=compress)
        self.db = ZODB.DB(compressed_storage, large_record_size=536870912)
        # self.open()
        # self.close_conn()

    def open(self):
        self.conn = self.db.open()
        self.root = self.conn.root()

    def insert(self, keys, values):
        with self.db.transaction() as connection:
            for key, value in zip(keys, values):
                self.root[key] = value
        # transaction.commit()
        return {'msg': f'record {keys}: {values} insert successfully.'}

    def commit(self):
        transaction.commit()

    def search(self, key):
        return self.root[key]

    def close_conn(self):
        self.conn.close()
        self.db.close()


DIC_ZODB = {
    'dependency': ZodbFactory('hub/dbs/ke_dependency.fs'),
    'spo': ZodbFactory('hub/dbs/ke_spo.fs'),
}
