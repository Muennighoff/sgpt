from .base import BaseEvaluator
import re
import os
import argparse
import tqdm
from typing import Dict, List, Set
from sklearn.metrics import average_precision_score, ndcg_score
from scipy.stats import pearsonr, spearmanr
import logging
import numpy as np
import pickle
import json
import torch
import math
from torch.nn import functional as F
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class Example(object):

    def __init__(self, s1, s2, is_para, score=None):
        self.s1 = s1
        self.s2 = s2
        self.is_para = is_para
        self.score = score

class TwitterURLData(object):

    def __init__(self, datasets_dir):
        assert 'Twitter_URL_Corpus_test.txt' in os.listdir(datasets_dir)
        self.data = []
        with open(os.path.join(datasets_dir, 'Twitter_URL_Corpus_test.txt'), 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                s1, s2, label = items[0], items[1], items[2]
                label = eval(label)[0]
                score = label * 20
                if label == 3:
                    is_para = None
                elif label > 3:
                    is_para = 1
                else:
                    assert label < 3
                    is_para = 0
                self.data.append(Example(s1, s2, is_para, score))


class PITData(object):

    def __init__(self, datasets_dir):
        assert 'test.data' in os.listdir(datasets_dir)
        assert 'test.label' in os.listdir(datasets_dir)
        self.data = []
        with open(os.path.join(datasets_dir, 'test.data'), 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                s1, s2, label = items[2], items[3], int(items[4])
                score = label * 20
                if label == 3:
                    is_para = None
                elif label > 3:
                    is_para = 1
                else:
                    assert label < 3
                    is_para = 0
                self.data.append(Example(s1, s2, is_para, score))


class TwitterParaEvaluator(BaseEvaluator):
    name = 'twitterpara'
    main_metric = 'ap_twitter_avg'

    def __init__(self, semb_fn, dataset_dir='data-eval/twitterpara', dname='all', bsz=32, show=True):
        BaseEvaluator.__init__(self, semb_fn, bsz, show)
        assert 'Twitter_URL_Corpus_test.txt' in os.listdir(dataset_dir)
        assert 'test.data' in os.listdir(dataset_dir)
        assert 'test.label' in os.listdir(dataset_dir)
        twitterurl_dir = pit_dir = dataset_dir
        self.dataset_dict = {
            'twitterurl': TwitterURLData(twitterurl_dir).data,
            'pit': PITData(pit_dir).data
        }
        assert dname in ['all', 'twitterurl', 'pit']
        if dname == 'all': self.dnames = ['twitterurl', 'pit']
        else: self.dnames = [dname]
    
    @property
    def metric_names(self):
        mnames = [f'ap_twitter_{dname}' for dname in self.dnames] + \
            [f'spearman_twitter_{dname}' for dname in self.dnames]
        if len(self.dnames) > 1: 
            mnames.extend([
                'ap_twitter_avg', 
                'spearman_twitter_avg'
            ])
        return mnames

    def _run(self, eval_type=None, normalize=True):
        if eval_type == 'valid':
            logging.warning('TwitterPara does not have a development set and here the model is evaluated on the test set instead.')
        results = {}
        for dname in self.dnames:
            dataset = self.dataset_dict[dname]
            is_para = [e.is_para for e in dataset]
            gold_scores = [e.score for e in dataset]
            s1s = [e.s1 for e in dataset]
            s2s = [e.s2 for e in dataset]
            s1_embs = self._text2se(s1s, normalize=normalize, add_name=f"{dname}1")  # (bsz, hdim)
            s2_embs = self._text2se(s2s, normalize=normalize, add_name=f"{dname}2")  # (bsz, hdim)
            pred_scores = F.cosine_similarity(s1_embs, s2_embs, dim=-1).cpu().numpy()
            not_none = [i for i, l in enumerate(is_para) if l is not None]
            is_para_not_none = list(np.array(is_para)[not_none])
            pred_scores_not_none = list(pred_scores[not_none])
            ap = average_precision_score(is_para_not_none, pred_scores_not_none)
            corr = spearmanr(gold_scores, pred_scores)
            results[f'ap_twitter_{dname}'] = ap
            results[f'spearman_twitter_{dname}'] = corr.correlation
        if len(self.dnames) > 1:
            results['ap_twitter_avg'] = np.mean([v for k, v in results.items() if 'ap_twitter' in k])
            results['spearman_twitter_avg'] = np.mean([v for k, v in results.items() if 'spearman_twitter_' in k])
        return results
