# -*- coding: utf-8 -*-
import pandas as pd
import json
from pymongo import MongoClient

settings = {
        'mongo_host': 'server.local',
        'mongo_db_name': 'mydb',
        'mongo_port': 27017,
        'chunk_size': 100000,
        'drop_collections_on_load': True,
        'transactions_collection': 'transactions',
        'transactions_source_csv_gz': 'transactions.csv.gz'
    }


def to_mongo(dest, data, idxs=[]):
    if (settings['drop_collections_on_load']):
        dest.drop()
    for chunk in data:
        dest.insert(json.loads(chunk.to_json(orient='records')))
    for idx in idxs:
        dest.ensure_index(idx)

if __name__ == "__main__":
    # establish connection:
    mongo_client = MongoClient(settings['mongo_host'],
                               settings['mongo_port'])
    mongo_db = mongo_client[settings['mongo_db_name']]

    # load data:
    transactions = pd.read_csv(
        settings['transactions_source_csv_gz'],
        parse_dates=['date'],
        compression='gzip',
        chunksize=settings['chunk_size'])

    # insert data:
    to_mongo(mongo_db[settings['transactions_collection']],
             transactions,
             ['id', 'brand', 'category', 'company', 'date'])

    # close connection:
    mongo_client.close()
