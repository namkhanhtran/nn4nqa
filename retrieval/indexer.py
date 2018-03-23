#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import argparse
import pandas as pd
from retrieval.lucene_tools import Lucene


def process_index(args):
    """"""
    indexer = Lucene(index_dir=args.index_path)
    indexer.open_writer()

    df = pd.read_csv(args.data_path, sep="\t", usecols=['docid', 'doc', 'timestamp'])
    df = df.dropna()
    for item in df.values:
        doc = {'id': str(item[0]), 'content': item[1], 'date': item[2]}

        document = []
        id_field = dict()
        id_field['field_name'] = 'id'
        id_field['field_value'] = doc['id']
        id_field['field_type'] = 'id'
        document.append(id_field)

        content_field = dict()
        content_field['field_name'] = 'content'
        content_field['field_value'] = indexer.preprocess(doc['content'])
        content_field['field_type'] = "text_tvp"
        document.append(content_field)

        date_field = dict()
        date_field['field_name'] = 'date'
        date_field['field_value'] = doc['date']
        date_field['field_type'] = 'id'
        document.append(date_field)

        indexer.add_document(document)

    indexer.close_writer()


#############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='/path/to/data')
    parser.add_argument('--index_path', type=str, help='/path/to/saved/index')
    args = parser.parse_args()

    process_index(args)
