#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-25 16:10:44
# @Author  : Joe Gao (jeusgao@163.com)

import os
import time
from milvus import Milvus, IndexType, MetricType, Status

dim = 1024


class MilvusFactory(object):
    def __init__(self, host='127.0.0.1', port='9058'):
        self.db = Milvus(host, port)

    def build_collection(self, collection_name='dependency', partition_tag='202103', dim=dim, index_file_size=1024):
        param = {
            'collection_name': collection_name,
            'dimension': dim,
            'index_file_size': index_file_size,
            'metric_type': MetricType.IP
        }
        self.db.create_collection(param)

        self.db.create_partition(collection_name=collection_name, partition_tag=partition_tag)
        self.db.list_partitions(collection_name=collection_name)

        index_param = {"M": 32, "efConstruction": 256}
        self.db.create_index(collection_name, IndexType.HNSW, index_param)

    def insert(self, vecs, collection_name='dependency', partition_tag='202103', dim=dim, index_file_size=1024):
        _, ok = self.db.has_collection(collection_name)
        if not ok:
            self.build_collection(
                collection_name=collection_name,
                partition_tag=partition_tag,
                dim=dim,
                index_file_size=index_file_size
            )
        status, ids = self.db.insert(collection_name, vecs, partition_tag=partition_tag)
        self.db.flush([collection_name])

        return ids

    def search(self, query_vectors, collection_name='dependency', partition_tags=None, top_k=5):
        search_param = {
            "ef": 16384,
            # "nprobe": 8192
            # "search_k": top_k * 768
        }

        st = time.time()
        param = {
            'collection_name': collection_name,
            'query_records': query_vectors,
            'top_k': top_k,
            'params': search_param,
        }
        if partition_tags:
            if isinstance(partition_tags, str):
                partition_tags = [partition_tags]
            param['partition_tags'] = partition_tags

        status, simi = self.db.search(**param)
        print(f'...Searching Cost: {time.time()-st}')
        if status.OK():
            return simi
        else:
            print("Search failed. ", status)
            return None


milvusDB = MilvusFactory()
