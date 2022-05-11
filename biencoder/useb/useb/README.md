# Preface

This is a clone of the USEB repo at the commit `17f013829092e3445da5ab79b1cc0fe9af95e2a4`.
Changes:
- Feed dataset & batchsize information into sentence embedding function to allow saving & caching embeddings
- Expose normalize as parameter


# Unsupervised Sentence Embedding Benchmark (USEB)
This repository hosts the data and the evaluation script for reproducing the results reported in the paper: "[TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979)". This benchmark (USEB) contains four heterogeous, task- and domain-specific datasets: [AskUbuntu](https://github.com/taolei87/askubuntu), [CQADupStack](https://github.com/D1Doris/CQADupStack), [TwitterPara](https://www.aclweb.org/anthology/D17-1126/) and [SciDocs](https://github.com/allenai/scidocs). It directly works with [SBERT](https://github.com/UKPLab/sentence-transformers). For details, pleasae refer to the paper.

## Install
```python
pip install useb  # Or git clone and pip install .
python -m useb.downloading all  # Download both training and evaluation data
```

## Usage & Example
After data downloading, one can either run (it needs ~8min on a GPU)
```bash
python -m useb.examples.eval_sbert
```
to evaluate an [SBERT](https://github.com/UKPLab/sentence-transformers) model (really an awesome repository for sentence embeddings, and the lastest model there is much better) on all the datasets; or run this same code below:
```python
from useb import run
from sentence_transformers import SentenceTransformer  # SentenceTransformer is an awesome library for providing SOTA sentence embedding methods. TSDAE is also integrated into it.
import torch

sbert = SentenceTransformer('bert-base-nli-mean-tokens')  # Build an SBERT model

# The only thing needed for the evaluation: a function mapping a list of sentences into a batch of vectors (torch.Tensor)
@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    return torch.Tensor(sbert.encode(sentences, show_progress_bar=False))

results, results_main_metric = run(
    semb_fn_askubuntu=semb_fn, 
    semb_fn_cqadupstack=semb_fn,  
    semb_fn_twitterpara=semb_fn, 
    semb_fn_scidocs=semb_fn,
    eval_type='test',
    data_eval_path='data-eval'  # This should be the path to the folder of data-eval
)

assert round(results_main_metric['avg'], 1) == 47.6
```
It is also supported to evaluate on a single dataset (please see [useb/examples/eval_sbert_askubuntu.py](useb/examples/eval_sbert_askubuntu.py)):
```bash
python -m useb.examples.eval_sbert_askubuntu
```

## Data Organization
```bash
.
├── data-eval  # For evaluation usage. One can refer to ./unsupse_benchmark/evaluators to learn about how to loading these data.
│   ├── askubuntu
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── text_tokenized.txt
│   ├── cqadupstack
│   │   ├── corpus.json
│   │   └── retrieval_split.json
│   ├── scidocs
│   │   ├── cite
│   │   │   ├── test.qrel
│   │   │   └── val.qrel
│   │   ├── cocite
│   │   │   ├── test.qrel
│   │   │   └── val.qrel
│   │   ├── coread
│   │   │   ├── test.qrel
│   │   │   └── val.qrel
│   │   ├── coview
│   │   │   ├── test.qrel
│   │   │   └── val.qrel
│   │   └── data.json
│   └── twitterpara
│       ├── Twitter_URL_Corpus_test.txt
│       ├── test.data
│       └── test.label
├── data-train  # For training usage.
│   ├── askubuntu
│   │   ├── supervised  # For supervised training. *.org and *.para are parallel files, each line are aligned and compose a gold relevant sentence pair (to work with MultipleNegativeRankingLoss in the SBERT repo).
│   │   │   ├── train.org
│   │   │   └── train.para
│   │   └── unsupervised  # For unsupervised training. Each line is a sentence.
│   │       └── train.txt
│   ├── cqadupstack
│   │   ├── supervised
│   │   │   ├── train.org
│   │   │   └── train.para
│   │   └── unsupervised
│   │       └── train.txt
│   ├── scidocs
│   │   ├── supervised
│   │   │   ├── train.org
│   │   │   └── train.para
│   │   └── unsupervised
│   │       └── train.txt
│   └── twitter  # For supervised training on TwitterPara, the float labels are also available (to work with CosineSimilarityLoss in the SBERT repo). As reported in the paper, using the float labels can achieve higher performance.
│       ├── supervised
│       │   ├── train.lbl
│       │   ├── train.org
│       │   ├── train.para
│       │   ├── train.s1
│       │   └── train.s2
│       └── unsupervised
│           └── train.txt
└── tree.txt
```

## Citation
If you use the code for evaluation, feel free to cite our publication [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979):
```bibtex 
@article{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and  Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2104.06979",
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2104.06979",
}
```

Contact person and main contributor: [Kexin Wang](https://kwang2049.github.io/), kexin.wang.2049@gmail.com

[https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

[https://www.tu-darmstadt.de/](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
