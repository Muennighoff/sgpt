from .base import BaseEvaluator
import re
import os
import argparse
import tqdm
from typing import Dict, List
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import logging
import numpy as np
import pickle
import json
import torch
import math
from torch.nn import functional as F
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class PoolExample(object):

    def __init__(self, qid, title, body):
        self.qid = qid
        self.title = title
        self.body = body

class EvalExample(object):

    def __init__(self, qid, gold_similar, bm25_rank):
        self.qid = qid
        self.gold_similar = gold_similar
        self.bm25_rank = bm25_rank

def rank_by_score(candidates, scores):
    sorted_kvs = sorted(list(zip(candidates, scores)), key=lambda *kv: kv[0][1], reverse=True)
    return [qid for qid, _ in sorted_kvs]

class AskUbuntuData(object):

    def __init__(self, datasets_dir):
        assert 'text_tokenized.txt' in os.listdir(datasets_dir)
        assert 'dev.txt' in os.listdir(datasets_dir)
        assert 'test.txt' in os.listdir(datasets_dir)
        self.pool = self._load_pool(os.path.join(datasets_dir, 'text_tokenized.txt'))
        self.dev = self._load_eval(os.path.join(datasets_dir, 'dev.txt'))
        self.test = self._load_eval(os.path.join(datasets_dir, 'test.txt'))
    
    def _load_pool(self, pool_path) -> Dict[str, PoolExample]:
        pool = {}
        with open(pool_path, 'r') as f:
            for line in f:
                qid, title, body = line.split('\t')
                pool[qid] = PoolExample(qid.strip(), title.strip(), body.strip())
        return pool

    def _load_eval(self, eval_path) -> List[EvalExample]:
        eval_examples = []
        with open(eval_path, 'r') as f:
            for line in f:
                qid, gold_similar, bm25_retrieved, bm25_scores = line.split('\t')
                gold_similar = list(gold_similar.split())
                bm25_retrieved = list(bm25_retrieved.split())
                bm25_scores = list(map(float, bm25_scores.strip().split()))
                bm25_rank = rank_by_score(bm25_retrieved, bm25_scores)
                eval_examples.append(EvalExample(qid, gold_similar, bm25_rank))
        return eval_examples

def reciprocal_rank(relevant_list, pred_list):
    assert len(set(relevant_list) & set(pred_list)) > 0
    rr = 1 / [i+1 for i, qid in enumerate(pred_list) if qid in relevant_list][0]
    return {'mrr': rr, 'rr': rr}

def ap_score(relevant_list, pred_list):
    assert len(relevant_list) > 0
    ap = []
    p_at_1 = None
    p_at_5 = None
    relevant_set = set(relevant_list)

    def precision_at_k(pred_to_k):
        return len(set(pred_to_k) & relevant_set) / len(pred_to_k)

    for idx, qid in enumerate(pred_list):
        k = idx + 1
        p_at_k = precision_at_k(pred_list[:k])
        if k == 1:
            p_at_1 = p_at_k
        if k == 5:
            p_at_5 = p_at_k
        if qid in relevant_set:
            ap.append(p_at_k)
    ap = np.mean(ap)

    return {'ap': ap , 'map': ap, 'p@1': p_at_1, 'p@5': p_at_5}

class AskubuntuEvaluator(BaseEvaluator, AskUbuntuData):
    name = 'askubuntu'
    main_metric = 'map_askubuntu_title'

    def __init__(self, semb_fn, datasets_dir='data-eval/askubuntu', text_components='title', bsz=32, show=True):
        BaseEvaluator.__init__(self, semb_fn, bsz, show)
        AskUbuntuData.__init__(self, datasets_dir)
        assert text_components in ['title_and_body', 'title', 'body']
        self.text_components = text_components
        self._metric_names = ['map', 'p@1', 'p@5', 'mrr']

    @property
    def metric_names(self):
        mnames = []
        for mname in self._metric_names:
            mnames.append(f'{mname}_askubuntu_{self.text_components}')
        return mnames

    def _get_sent(self, qid):
        if self.text_components == 'title_and_body':
            return ' '.join([self.pool[qid].title, self.pool[qid].body])
        elif self.text_components == 'title':
            return self.pool[qid].title
        else:
            assert self.text_components == 'body'
            return self.pool[qid].body

    def _run(self, eval_type, normalize=True):
        if eval_type == 'valid':
            eval_set = self.dev
        else:
            assert eval_type == 'test'
            eval_set = self.test
        
        result = {}
        show = bool(self.show)
        for eval_example in tqdm.tqdm(eval_set, disable=not self.show):
            qid = eval_example.qid
            gold_similar = eval_example.gold_similar
            if len(gold_similar) == 0:
                continue

            bm25_rank = eval_example.bm25_rank

            sents = [self._get_sent(qid),]
            for qid_candidate in bm25_rank:
                sents.append(self._get_sent(qid_candidate))
            
            self.show = False  # to mute the next line from showing progress bar
            embs = self._text2se(sents, normalize=normalize, add_name=f"{qid}")  # (bsz, hdim) = (1+20, hdim)
            qembs = embs[0:1]  # (1, hdim)
            cembs = embs[1:]  # (20, hdim)
            scores = torch.matmul(qembs, cembs.t()).squeeze(0).cpu().tolist()  # (1, 20)
            mdl_rank = rank_by_score(bm25_rank, scores)
            
            result_query = {}
            result_query.update(ap_score(gold_similar, mdl_rank))
            result_query.update(reciprocal_rank(gold_similar, mdl_rank))
            for mname in self._metric_names:
                result.setdefault(mname, [])
                result[mname].append(result_query[mname])

        result = {f'{k}_askubuntu_{self.text_components}': np.mean(v) for k, v in result.items()}      
        self.show = show
        return result






