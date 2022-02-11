from .base import BaseEvaluator
import os
import tqdm
from typing import Dict, List, Set
from sklearn.metrics import average_precision_score, ndcg_score
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer
import logging
import numpy as np
import json
import torch
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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self


class CQADupStackData(object):

    def __init__(self, datasets_dir):
        assert 'corpus.json' in os.listdir(datasets_dir)
        assert 'retrieval_split.json' in os.listdir(datasets_dir)
        with open(os.path.join(datasets_dir, 'corpus.json'), 'r') as f:
            self.corpus = json.load(f)
        with open(os.path.join(datasets_dir, 'retrieval_split.json'), 'r') as f:
            self.retrieval_split = json.load(f)
        # the forum should be one of:
        #     'android', 'english', 'gaming', 'gis', 
        #     'mathematica', 'physics', 'programmers', 
        #     'stats', 'tex', 'unix', 'webmasters', 
        #     'wordpress'
        self.forums = list(self.corpus.keys())
    
    def _get_text(self, forum, qid):
        return self.corpus[forum][qid]
    
    def get_forum_data(self, forum, eval_type):
        assert forum in self.forums
        assert eval_type in ['valid', 'test']
        qrels = self.retrieval_split[eval_type][forum]
        eval_queries = {qid: self._get_text(forum, qid) for qid in qrels}
        rel_docs = qrels
        rel_docs_set = {qid: set(dids) for qid, dids in qrels.items()}
        pool = dict(self.corpus[forum])
        [pool.pop(qid) for qid in qrels]  # remove all the queries, since we do not want to return the identical question to the query
        data_dict = {
            'eval_queries': eval_queries,
            'pool': pool,
            'rel_docs': rel_docs,
            'rel_docs_set': rel_docs_set
        }
        return AttrDict(**data_dict)


class CQADupStackEvaluator(BaseEvaluator, CQADupStackData):
    name = 'cqadupstack'
    main_metric = 'map@100_cqadupstack_avg'

    def __init__(self, semb_fn, datasets_dir='data-eval/cqadupstack', forum='all', bsz=32, show=True):
        BaseEvaluator.__init__(self, semb_fn, bsz, show)
        CQADupStackData.__init__(self, datasets_dir)        
        assert forum in set(self.forums) | {'all'}
        if forum == 'all':
            self.dnames = self.forums
        else:
            self.dnames = [forum]

    @property
    def metric_names(self):
        mnames = [f'map@100_cqadupstack_{forum}' for forum in self.dnames] + \
            [f'ndcg@10_cqadupstack_{forum}' for forum in self.dnames]
        if len(self.dnames) > 1:
            mnames.extend(['map@100_cqadupstack_avg', 'ndcg@10_cqadupstack_avg'])
        return mnames

    def compute_metrics(self, score_mtrx, qids, dids, rel_docs, rel_docs_set):
    
        def compute_dcg_at_k(relevances, k):
            dcg = 0
            for i in range(min(len(relevances), k)):
                dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
            return dcg

        avp_scores = []
        ndcg_scores = []
        map_k = 100
        ndcg_k = 10
        scores_topk, indices_topk = torch.Tensor(score_mtrx).topk(max(map_k, ndcg_k), dim=-1)
        for qid, scores, indices in zip(qids, scores_topk, indices_topk):
            hit_or_not = lambda x: 1 if x in rel_docs_set[qid] else 0

            # Average precision
            map_pred_scores = scores[0:map_k].cpu().tolist()
            map_true_scores = [hit_or_not(dids[indice]) for indice in indices[0:map_k]]
            avp = average_precision_score(map_true_scores, map_pred_scores) if 1 in map_true_scores else 0
            avp_scores.append(avp)

            # NDCG scores
            predicted_relevance = [hit_or_not(dids[indice]) for indice in indices[0:ndcg_k]]
            true_relevances = [1] * len(rel_docs[qid])

            if len(true_relevances) > 0:
                ndcg_value = compute_dcg_at_k(predicted_relevance, ndcg_k) / compute_dcg_at_k(true_relevances, ndcg_k)
                ndcg_scores.append(ndcg_value)
        
        mean_ap_score = np.mean(avp_scores)
        mean_ndcg_score = np.mean(ndcg_scores)
        return mean_ap_score, mean_ndcg_score

    def _run(self, eval_type, normalize=True):
        results = {}
        for forum in self.dnames:
            fdata = self.get_forum_data(forum, eval_type)
            eval_queries, pool, rel_docs, rel_docs_set = \
                fdata.eval_queries, fdata.pool, fdata.rel_docs, fdata.rel_docs_set

            qids = []
            queries = []
            for qid, query in eval_queries.items():
                qids.append(qid)
                queries.append(query)
            qembs = self._text2se(queries, normalize=normalize, add_name=f"{forum}queries")
            
            dids = []
            docs = []
            for did, doc in pool.items():
                dids.append(did)
                docs.append(doc)
            dembs = self._text2se(docs, normalize=normalize, add_name=f"{forum}docs")
            
            score_mtrx = torch.matmul(qembs, dembs.t()).cpu().numpy()  # (nq, nd)
            mean_ap_score, mean_ndcg_score = self.compute_metrics(
                score_mtrx, 
                qids, 
                dids, 
                rel_docs, 
                rel_docs_set
            )
            results[f'map@100_cqadupstack_{forum}'] = mean_ap_score
            results[f'ndcg@10_cqadupstack_{forum}'] = mean_ndcg_score
            if self.show: logging.info(f'map/ndcg for {forum}: {mean_ap_score}/{mean_ndcg_score}')

        if len(self.dnames) > 1:
            map_avg = np.mean([v for k, v in results.items() if 'map@100_cqadupstack' in k])
            ndcg_avg = np.mean([v for k, v in results.items() if 'ndcg@10_cqadupstack' in k])
            results[f'map@100_cqadupstack_avg'] = map_avg
            results[f'ndcg@10_cqadupstack_avg'] = ndcg_avg
            if self.show: logging.info(f'map/ndcg for average: {map_avg}/{ndcg_avg}')
        return results




