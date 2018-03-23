"""
Console application for general-purpose retrieval.

first pass: get top N documents using Lucene's default retrieval method (based on the catch-all content field)
second pass: perform (expensive) scoring of the top N documents using the Scorer class

General config parameters:
- index_dir: index directory
- query_file: query file (JSON)
- model: accepted values: lucene, lm, mlm, prms (default: lm)
- output_file: output file name
- output_format: (default: trec) -- not used yet
- run_id: run in (only for "trec" output format)
- num_docs: number of documents to return (default: 100)
- field_id: id field to be returned (default: Lucene.FIELDNAME_ID)
- first_pass_num_docs: number of documents in first-pass scoring (default: 10000)
- first_pass_field: field used in first pass retrieval (default: Lucene.FIELDNAME_CONTENTS)

Model-specific parameters:
- smoothing_method: jm or dirichlet (lm and mlm, default: jm)
- smoothing_param: value of lambda or alpha (jm default: 0.1, dirichlet default: average field length)
- field_weights: dict with fields and corresponding weights (only mlm)
- field: field name for LM model
- fields: fields for PRMS model


@author: Krisztian Balog (krisztian.balog@uis.no)
"""
from datetime import datetime

import argparse
import os
import pandas as pd
from retrieval.lucene_tools import Lucene
from retrieval.scorer import Scorer
from retrieval.results import RetrievalResults
from tqdm import tqdm

import pickle

import numpy as np


class Retrieval(object):
    def __init__(self, args, lucene_vm_init=True):
        """
        Loads config file, checks params, and sets default values.

        """
        self.config = {'index_dir': args.index_dir,
                       'model': args.model,  # bm25 or lm
                       'smoothing_method': args.smoothing_method,  # jm or dirichlet
                       'first_pass_field': args.first_pass_field,
                       'field_id': args.field_id,
                       'first_pass_num_docs': args.first_pass_num_docs,
                       'num_docs': args.num_docs,
                       'query_file': args.query_file,
                       'output_file': args.output_file}

        # self._open_index(lucene_vm_init=lucene_vm_init)

    def _open_index(self, lucene_vm_init=True):
        self.lucene = Lucene(self.config['index_dir'], lucene_vm_init=lucene_vm_init)

        self.lucene.open_searcher()
        if self.config['model'] == 'lm':
            self.lucene.set_lm_similarity_jm(self.config['smoothing_method'])

    def _close_index(self):
        self.lucene.close_reader()

    def _first_pass_scoring(self, lucene, query):
        """
        Returns first-pass scoring of documents.

        :param query: raw query
        :return RetrievalResults object
        """
        # print("\tFirst pass scoring... ")
        results = lucene.score_query(query, field_content=self.config['first_pass_field'],
                                     field_id=self.config['field_id'],
                                     num_docs=self.config['first_pass_num_docs'])
        # print(results.num_docs())
        return results

    @staticmethod
    def _second_pass_scoring(res1, scorer):
        """
        Returns second-pass scoring of documents.

        :param res1: first pass results
        :return: RetrievalResults object
        """
        print("\tSecond pass scoring... ")
        results = RetrievalResults()
        for doc_id, orig_score in res1.get_scores_sorted():
            doc_id_int = res1.get_doc_id_int(doc_id)
            score = scorer.score_doc(doc_id, doc_id_int)
            results.append(doc_id, score)
        print("done")
        return results

    def query(self, query):
        query = Lucene.preprocess(query)
        if query == '':
            return None

        res1 = self._first_pass_scoring(self.lucene, query)

        scorer = Scorer.get_scorer(self.config['model'], self.lucene, query, self.config)
        result = self._second_pass_scoring(res1, scorer)

        return result

    def _load_queries(self):
        df = pd.read_csv(self.config['query_file'], sep="\t", usecols=['qid', 'question', 'timestamp'])
        df = df.dropna()

        valid = pickle.load(open("data/valid_sample.pkl", 'rb'))
        valid_queries = set([q for q, d in valid])

        self.queries = dict()
        for item in df.values:
            if item[0] in valid_queries:
                self.queries[str(item[0])] = item[1]

    def _load_test_queries(self):
        df = pd.read_csv(self.config['query_file'], sep="\t", usecols=['qid', 'question'])
        df = df.dropna()

        self.queries = dict()
        for item in df.values:
            self.queries[item[0]] = item[1]

    def _load_train_queries(self):
        df = pd.read_csv(self.config['query_file'], sep="\t", usecols=['qid', 'question', 'timestamp'])
        df = df.dropna()

        train = pickle.load(open("data/train_sample.pkl", 'rb'))
        train_queries = set([q for q, d in train])

        self.train_triples = dict()
        for q, d in train:
            if q not in self.train_triples:
                self.train_triples[q] = set()
            self.train_triples[q].add(d)

        self.queries = dict()
        for item in df.values:
            if item[0] in train_queries:
                self.queries[item[0]] = item[1]

        df_doc = pd.read_csv("data/FiQA_train_doc_final.tsv", sep="\t", usecols=['docid', 'doc', 'timestamp'])
        df_doc = df_doc.dropna()

        self.all_docs = set(df_doc['docid'].values)

    def generate_negative(self):
        """"""
        self._load_train_queries()
        print("Number of queries:", len(self.queries))
        self._open_index(lucene_vm_init=False)

        # init output file
        train_dataset = dict()

        for query_id in tqdm(sorted(self.queries)):
            query = Lucene.preprocess(self.queries[query_id])

            response = self._first_pass_scoring(self.lucene, query)
            docids = [int(docid) for docid, _ in response.get_scores_sorted()]

            docids = np.random.choice(list(docids), min(100, len(docids)), replace=False)

            train_dataset[query_id] = []
            for docid in self.train_triples[query_id]:
                train_dataset[query_id].append((docid, 1))
            for docid in docids:
                if docid not in self.train_triples[query_id]:
                    train_dataset[query_id].append((docid, 0))

        # {'queryid': array(int(docid))}
        pickle.dump(train_dataset, open(self.config['output_file'], 'wb'))

    def retrieve(self):
        """Scores queries and outputs results."""
        s_t = datetime.now()  # start time
        total_time = 0.0

        # self._load_queries()
        self._load_test_queries()
        print("Number of queries:", len(self.queries))
        self._open_index(lucene_vm_init=False)

        # init output file
        if os.path.exists(self.config['output_file']):
            os.remove(self.config['output_file'])
        out = open(self.config['output_file'], "w")

        for query_id in tqdm(sorted(self.queries)):
            query = Lucene.preprocess(self.queries[query_id])
            # print("scoring [" + query_id + "] " + query)
            # first pass scoring
            res1 = self._first_pass_scoring(self.lucene, query)
            # second pass scoring (if needed)
            if self.config['model'] in ["bm25", "lm"]:
                results = res1
            else:
                scorer = Scorer.get_scorer(self.config['model'], self.lucene, query, self.config)
                results = self._second_pass_scoring(res1, scorer)
            # write results to output file
            results.write_trec_format(str(query_id), out, self.config['num_docs'])

        # close output file
        out.close()
        # close index
        self._close_index()

        e_t = datetime.now()  # end time
        diff = e_t - s_t
        total_time += diff.total_seconds()
        time_log = "Execution time(sec):\t" + str(total_time) + "\n"
        print(time_log)

    def process_qa(self):
        self._open_index(lucene_vm_init=False)
        df_doc = pd.read_csv("data/FiQA_train_doc_final.tsv", sep="\t", usecols=['docid', 'doc', 'timestamp'])
        df_doc = df_doc.dropna()

        df_question = pd.read_csv('data/FiQA_train_question_final.tsv', sep="\t",
                                  usecols=['qid', 'question', 'timestamp'])
        df_question = df_question.dropna()

        answers = dict()
        for docid, doc in tqdm(df_doc[['docid', 'doc']].values):
            answers[docid] = Lucene.preprocess(doc)

        questions = dict()
        for qid, question in tqdm(df_question[['qid', 'question']].values):
            questions[qid] = Lucene.preprocess(question)

        pickle.dump(questions, open('data/processed_questions_lucene.pkl', 'wb'))
        pickle.dump(answers, open('data/processed_answers_lucene.pkl', 'wb'))


def main(args):
    r = Retrieval(args)
    r.retrieve()
    # r.generate_negative()
    # r.process_qa()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, help='/path/to/index')
    parser.add_argument('--query_file', type=str, help='/path/to/query_file')
    parser.add_argument('--output_file', type=str, help='/path/to/output_file')
    parser.add_argument('--model', type=str, default='bm25', help='model: lm or lucene')
    parser.add_argument('--num_docs', type=int, default=1000, help='number of docs')
    parser.add_argument('--field_id', type=str, default='id', help='field id')
    parser.add_argument('--first_pass_num_docs', type=int, default=1000, help='number of docs in first pass')
    parser.add_argument('--first_pass_field', type=str, default='content', help='field for first pass')
    parser.add_argument('--smoothing_method', type=str, default='jm', help='smoothing methods')

    args = parser.parse_args()

    main(args)
