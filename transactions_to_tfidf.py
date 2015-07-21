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
from tfidf_module import *

import numpy as np
import pandas as pd
import json
from pymongo import MongoClient, ASCENDING

settings = {
        'mongo_host': 'server.local',
        'mongo_db_name': 'mydb',
        'mongo_port': 27017,
        'drop_collections_on_load': True,
        'transactions_collection': 'transactions',
        'tfidf_collection': 'tfidf'
    }

if __name__ == "__main__":
    # establish connection:
    mongo_client = MongoClient(settings['mongo_host'],
                               settings['mongo_port'])
    mongo_db = mongo_client[settings['mongo_db_name']]

    # get collections:
    transactions_collection = mongo_db[settings['transactions_collection']]
    tfidf_collection = mongo_db[settings['tfidf_collection']]
    if (settings['drop_collections_on_load']):
        tfidf_collection.drop()

    # query number of consumers d in cohort D
    pipeline = [
            {"$group": {
                "_id": "$id"
            }},
            {"$group": {
                "_id": "null",
                "count": {"$sum": 1}
            }},
            {"$project": {
                "_id": False,
                "count": True
            }}
        ]
    cursor = transactions_collection.aggregate(pipeline,
                                               allowDiskUse=True)
    result = list(cursor)
    n_consumers = result[0]['count']

    # query items:
    pipeline = [
            {"$group": {
                "_id": {
                    "brand": "$brand",
                    "company": "$company",
                    "category": "$category"
                }
            }},
            {"$project": {
                "_id": False,
                "brand": "$_id.brand",
                "company": "$_id.company",
                "category": "$_id.category"
            }}
        ]
    cursor = transactions_collection.aggregate(pipeline,
                                               allowDiskUse=True)
    items = pd.DataFrame(list(cursor))
    items.index.name = "item"
    items = items.reset_index(level=0)
    items = items.set_index(['brand',
                             'company',
                             'category'])

    # get items t
    # (group by brand, company, category)
    pipeline = [
            {"$group": {
                "_id": {
                    "consumer": "$id",
                    "brand": "$brand",
                    "company": "$company",
                    "category": "$category"
                },
                "purchasecount": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$gt": [
                                    "$purchasequantity",
                                    0
                                ]},
                                {"$gt": [
                                    "$purchaseamount",
                                    0
                                ]}
                            ]},
                            "$purchasequantity",
                            0
                        ]
                    }
                }
            }},
            {"$group": {
                "_id": {
                    "brand": "$_id.brand",
                    "company": "$_id.company",
                    "category": "$_id.category"
                },
                "consumers": {"$push": "$_id.consumer"},
                "purchasecounts": {"$push": "$purchasecount"}
            }},
            {"$project": {
                "_id": False,
                "brand": "$_id.brand",
                "company": "$_id.company",
                "category": "$_id.category",
                "consumers": True,
                "purchasecounts": True
            }}
        ]
    cursor = transactions_collection.aggregate(pipeline,
                                               allowDiskUse=True)
    # process records:
    for record in cursor:
        df = pd.DataFrame(record['consumers'],
                          columns=['consumer'])
        df.loc[:, 'item'] = items.loc[(record['brand'],
                                       record['company'],
                                       record['category']),
                                      'item']
        df.loc[:, 'brand'] = record['brand']
        df.loc[:, 'company'] = record['company']
        df.loc[:, 'category'] = record['category']
        df.loc[:, 'purchasecount'] = record['purchasecounts']
        df = df.join(
                pd.DataFrame(
                    get_tfidf(
                        np.array(
                            df['purchasecount'].values,
                            dtype=np.int64),
                        n_consumers
                    ),
                    columns=['purchasetfidf1',
                             'purchasetfidf2']
                )
            )
        # insert data:
        _ = tfidf_collection.insert(
                json.loads(
                    df.to_json(orient='records')
                )
            )

    # create indexes:
    tfidf_collection.create_index(
        [('consumer', ASCENDING)],
        background=True)
    tfidf_collection.create_index(
        [('item', ASCENDING)],
        background=True)
    tfidf_collection.create_index(
        [('brand', ASCENDING),
         ('company', ASCENDING),
         ('category', ASCENDING)],
        background=True)

    # close connection to MongoDB
    mongo_client.close()
