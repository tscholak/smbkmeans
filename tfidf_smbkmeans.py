# -*- coding: utf-8 -*-
import os
import sys
import inspect
cmd_folder = os.path.realpath(
        os.path.abspath(
            os.path.split(
                inspect.getfile(
                    inspect.currentframe()
                )
            )[0]
        )
    )
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
from smbkmeans import *

import pandas as pd
import numpy as np
import scipy.sparse as sp
import random

from bson.son import SON
from pymongo import MongoClient
from monary import Monary

import bz2
try:
    import cPickle as pickle
except:
    import pickle

settings = {
        'mongo_host': 'server.local',
        'mongo_db_name': 'mydb',
        'mongo_port': 27017,
        'tfidf_collection': 'tfidf',
        'models_per_k': 25,
        'ld_k_min': 0.5,
        'ld_k_max': 2.5,
        'k_steps': 50,
        'batch_size': 1024
    }

blacklist = {
        'consumers': [],
        'brands': [0],
        'companies': [10000],
        'categories': [0]
    }

if __name__ == "__main__":
    # establish PyMongo connection:
    mongo_client = MongoClient(settings['mongo_host'],
                               settings['mongo_port'])
    mongo_db = mongo_client[settings['mongo_db_name']]

    # get collection:
    tfidf_collection = mongo_db[settings['tfidf_collection']]

    # find out who the consumers are
    cursor = tfidf_collection.find(
            {"consumer": {
                "$nin": blacklist['consumers']
            }}
        ).distinct('consumer')
    consumers = np.array(cursor, dtype=np.int64)
    n_consumers = len(consumers)

    # find out how many items there are
    cursor = tfidf_collection.find().distinct('item')
    items = np.array(cursor, dtype=np.int64)
    n_items = len(items)

    # close PyMongo connection
    mongo_client.close()

    # set up Monary
    monary_client = Monary(settings['mongo_host'],
                           settings['mongo_port'])

    def get_consumer_mtx(consumer_batch):
        '''Returns a sparse matrix with feature vectors for a consumer batch.'''
        pipeline = [
                {"$match": {
                    "consumer": {"$in": consumer_batch},
                    "brand": {"$nin": blacklist['brands']},
                    "company": {"$nin": blacklist['companies']},
                    "category": {"$nin": blacklist['categories']}
                }},
                {"$project": {
                    "_id": False,
                    "consumer": True,
                    "item": True,
                    "tfidf": "$purchasetfidf2"
                }},
                {"$sort": SON([("consumer", 1)])}
            ]
        try:
            # careful! Monary returns masked numpy arrays!
            result = monary_client.aggregate(
                    settings['mongo_db_name'],
                    settings['tfidf_collection'],
                    pipeline,
                    ["consumer", "item", "tfidf"],
                    ["int64", "int64", "float64"])
        except:
            return sp.csr_matrix(shape=(len(consumer_batch), n_items),
                                 dtype=np.float64)

        # convert into CSR matrix
        _, consumer_idcs = np.unique(result[0].data,
                                     return_inverse=True)
        mtx = sp.csr_matrix(
                (result[2].data, (consumer_idcs,
                                  result[1].data)),
                shape=(len(consumer_batch), n_items),
                dtype=np.float64)

        # normalize each row (this step can't be moved into the database
        # because of the item blacklist)
        for row_idx in xrange(len(consumer_batch)):
            row = mtx.data[mtx.indptr[row_idx]:mtx.indptr[row_idx + 1]]
            row /= np.linalg.norm(row)

        return mtx

    def get_batch(batch_size=100, offset=0, random_pick=True):
        if random_pick:
            # pick batch_size examples randomly from the consumers in the
            # collection
            consumer_batch = random.sample(consumers, batch_size)
        else:
            # advance index by offset
            consumer_batch = list(consumers)[offset:]
            # get the next batch_size consumers from the collection
            consumer_batch = consumer_batch[:batch_size]

        # obtain sparse matrix filled with feature vectors from database
        mtx = get_consumer_mtx(consumer_batch)

        return mtx

    # train the models
    ns_clusters = np.unique(np.int64(np.floor(
            10. ** np.linspace(settings['ld_k_min'],
                               settings['ld_k_max'],
                               settings['k_steps'],
                               endpoint=True))))
    np.random.shuffle(ns_clusters)
    ns_clusters = ns_clusters.tolist()
    models = [SphericalMiniBatchKMeans(n_clusters=n_clusters,
                                       n_init=10,
                                       max_iter=1000,
                                       batch_size=settings['batch_size'],
                                       reassignment_ratio=.01,
                                       max_no_improvement=10,
                                       project_l=5.) for _ in xrange(settings['models_per_k']) for n_clusters in ns_clusters]
    filename = cmd_folder + '/tfidf_smbkmeans__tfidf2.pkl.bz2'
    for model in models:
        _ = model.fit(n_samples=n_consumers,
                      get_batch=get_batch)
        fp = bz2.BZ2File(filename, 'w')
        pickle.dump(models, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()
