"""
Setup:
!mkdir /home/.kaggle/
!mv kaggle.json /home/.kaggle/kaggle.json
!kaggle datasets download -d 'muennighoff/beirbm25results'
!unzip beirbm25results
!pip install -q beir transformers accelerate
"""

from beir import util, LoggingHandler
import logging
# Code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

import argparse
import collections
import json
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking import Rerank

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--modelpath", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    return args

args = parse_args()

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

model_path = args.modelpath
model_out_name = model_path.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=getattr(torch, args.dtype),
    max_memory=get_gpus_max_memory("50GB"),
    offload_folder="offload",
)
use_custom_model = True
batch_size = args.batchsize
debug = False
beir_data_dir = args.datadir

# All datasets
datasets = ["trec-covid", "webis-touche2020", "nfcorpus", "scifact", "fiqa", "dbpedia-entity",
            "nq", "hotpotqa", "quora", "fever", "climate-fever", "arguana", "msmarco", "scidocs", "cqadupstack",
            "signal1m", "trec-news", "bioasq", "robust04"]
# Comment this out if you wish to run all
datasets = [args.dataset]

# Main prompt
prompts = {"G": 'Documents are searched to find matches with the same content.\nThe document "{}" is a good search result for "',}


def encode(requests, tokenizer):
    new_reqs = []
    # Changed the order from original; as requests is queries, docs & we want query to be the continuation
    for continuation, context in requests:
        if context == "":
            # end of text as context
            context_enc = [tokenizer.eos_token_id]
        else:
            context_enc = tokenizer.encode(context, add_special_tokens=False)

        continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)

        new_reqs.append(((context, continuation), context_enc, continuation_enc))

    return new_reqs

def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)
    
    return list(res.values())


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [
            ([y[0] for y in x], x[0][1]) for x in arr
        ]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr
        
    
    def get_reordered(self):
        return [x[1] for x in self.arr]
    
    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds: 
                res[ind] = v
                cov[ind] = True
        
        assert all(cov)
        
        return res

def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []
    
    if arr: yield arr

def _model_call(inps, model):
    """
    inps: a torch tensor of shape [batch, sequence]
    the size of sequence may vary from call to call
    returns: a torch tensor of shape [batch, sequence, vocab] with the
    logits retuned from the model
    """
    return model(inps)[0][:, :, :].to(torch.float32)

def _loglikelihood_tokens(requests, model, max_length, device, disable_tqdm=False, batch_size=1, 
                          sub_select_idx=None, instruction_len=0, tokenizer=None, debug=False):
    # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
    res = []
    with torch.no_grad():

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
            #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return (-len(toks), tuple(toks))
        
        # TODO: automatic (variable) batch size detection for vectorization
        reord = Reorderer(requests, _collate)
        for chunk in chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), batch_size):
            inps = []
            contlens = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= max_length, f"Got {len(continuation_enc)} but max len is only {max_length}"

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                # cont_toks      4 5 6 7 8 9

                # Original:
                #inp = torch.tensor(
                #    (context_enc + continuation_enc)[-(self.max_length+1):][:-1]
                #, dtype=torch.long).to(self.device)
                
                # Modified (Muennighoff)
                # when too long to fit in context, truncate from the left & remove fin token # NM: + After the initial instruction
                inp = torch.tensor(
                    # Instruction + Text + Continuation
                    # Truncation from right: [:(max_length+1-instruction_len)]
                    # Truncation from left:  [-(max_length+1-instruction_len):]
                    #(context_enc[:instruction_len] + ((context_enc[instruction_len:] + continuation_enc)[:(max_length+1-instruction_len)]))[:-1]
                    (context_enc[:instruction_len] + ((context_enc[instruction_len:] + continuation_enc)[-(max_length+1-instruction_len):]))[:-1]
                , dtype=torch.long).to(device)
                inplen, = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad to length
                inp = torch.cat([
                    inp, # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device) # [padding_length - seq]
                ], dim=0)

                if debug:
                    print("Model Input")
                    print(tokenizer.decode(inp))

                inps.append(inp.unsqueeze(0))
                contlens.append(cont)
                inplens.append(inplen)
               
            if sub_select_idx:
                if debug:
                    print("Subselecting tokens:")
                    print(tokenizer.decode(sub_select_idx))
                # Subselect vocab for softmax by masking out all other vocab
                mask = torch.zeros_like(output_logits)
                mask[:,:,sub_select_idx] = 1
                output_logits = output_logits.masked_fill(mask == 0, float('-inf'))
                multi_logits = F.log_softmax(output_logits, dim=-1).cpu()
            else:
                multi_logits = F.log_softmax(_model_call(torch.cat(inps, dim=0), model), dim=-1).cpu()  # [batch, seq, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                contlen = len(cont_toks)

                logits = logits[inplen-contlen:inplen].unsqueeze(0) # [1, seq, vocab]

                greedy_tokens = logits.argmax(dim=-1)

                # cont_toks :: [1, seq]
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                
                if debug:
                    print("Continuation Given")
                    print(tokenizer.batch_decode(cont_toks))
                    print("Continuation Produced")
                    print(tokenizer.batch_decode(greedy_tokens))

                # cont_toks are the vocab indices that make up the perfect continuation
                # Hence we gather those vocab indices from the logits, i.e. their probabilities
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]

                # Sum to get a total score of that continuation
                res.append(float(logits.sum()))

    return reord.get_original(res)


class GPTRanker:
    def __init__(self, model=None, model_path="", use_prompt=True, prompt_doc="{}\n", prompt_doc_start="{}\n{}\n",
                 debug=False, fewshots="", **kwargs):
        """
        GPTRanker producing log-probabilities for reranking doc & query with a GPT-like model
        Args:
            model_path: HuggingFace weight name of a decoder transformer model
            use_prompt: Whether to use a prompt
            prompt_doc: Prompting scheme to embed document and query 
                Needs to contain two {} as query is not used for logprobs in this ranker
            prompt_doc_start: Prompting scheme specifically used for the first example, e.g. to include description
            fewshots: Fewshot example to use [doc, query]
            debug: To get information while running
        """
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        if model is not None:
            # Allow specific model loads, e.g. for GPT-J half-precision; Or parallel
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.model.config.model_type == 'gpt2':
            self.max_length = self.model.config.n_ctx
        elif self.model.config.model_type == 'gptj':
            self.max_length = self.model.config.n_ctx
        elif self.model.config.model_type == 'gpt_neo':
            self.max_length = self.model.config.max_position_embeddings
        elif self.model.config.model_type == 'bloom':
            self.max_length = self.model.config.seq_length
        else:
            raise ValueError(f"Unknown model of type {self.model.config.model_type}")
            
        # Truncation will be done from the left in the log likelihood
        self.prompt_doc = prompt_doc
        self.use_prompt = use_prompt
        self.instruction_len = len(self.tokenizer.tokenize(self.prompt_doc[:self.prompt_doc.index("{")]))
        self.debug = debug
    
        self.fewshots = fewshots
        if self.fewshots:
            # doc, query
            self.fewshots = prompt_doc_start.format(self.fewshots[0], self.fewshots[1])
            # Still take overflowing tokens away from the current doc (not the fewshot doc)
            self.instruction_len += len(self.tokenizer.tokenize(self.fewshots))
            
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        """
        Args:
          sentences: [query, document]
          batch_size: Unused

        Returns:
          log_probs: float log probability for each query-doc pair
        """
        # TODO: Possibly feed in batched?; Depending on model size?
        if self.use_prompt:
            # Leave queries as is, as all its tokens will be used to compute the loglikelihoods
            sentences = [(query, self.fewshots + self.prompt_doc.format(doc)) for (query, doc) in sentences]

        encoded = encode(sentences, self.tokenizer)
        # loglikelihood batch_size is not the batch_size fed into this func
        log_probs = _loglikelihood_tokens(encoded, self.model, self.max_length, self.device, instruction_len=self.instruction_len, 
                                          tokenizer=self.tokenizer, debug=self.debug)

        return log_probs


def clean_titles(corpus):
    for k in corpus:
        if "title" in corpus[k] and corpus[k]["title"] is None:
            corpus[k]["title"] = ""
    return corpus


def run_reranking(results_bm25_path, results_path, data_path, top_k=100, k_values=[1, 3, 5, 10, 100, 1000]):
    """
    Args:
        results_bm25_path: Path to .json results from bm25 for the dataset
        results_path: Path to .json to write rerank results
        top_k: How many docs to rerank per query
        k_values: For how many docs per query to compute the scores
    """
    
    split = "dev" if "msmarco" in data_path else "test"
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    
    corpus = clean_titles(corpus) if "robust04" in data_path else corpus
    
    with open(results_bm25_path, 'r') as fp:
        results_bm25 = json.load(fp)
    
    # Optional, make sure results are correct
    ndcg_bm25, _map_bm25, recall_bm25, precision_bm25 = EvaluateRetrieval.evaluate(qrels, results_bm25, k_values)

    # Rerank top-100 results using the reranker provided
    results_rerank = reranker.rerank(corpus, queries, results_bm25, top_k=top_k)
    
    # Save rerank results
    with open(results_path, 'w') as fp:
        json.dump(results_rerank, fp)

    #### Evaluate retrieval using NDCG@k, MAP@K ...
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_rerank, k_values)

    return (ndcg_bm25, _map_bm25, recall_bm25, precision_bm25), (ndcg, _map, recall, precision)

for prompt_id, prompt_doc in prompts.items():
    
    scores_out_path = f"beir_scores_{model_out_name}_{prompt_id}.json"
    
    # Optionally skip
    #if os.path.exists(os.path.join(os.getcwd(), scores_out_path)):
    #    continue

    ndcgs_bm25 = {}
    ndcgs = {}
    
    logging.info(f"\n{'-' * 20} Running prompt {prompt_id}: {prompt_doc} {'-' * 20}\n")
    
    if use_custom_model:
        reranker = Rerank(GPTRanker(model=model, model_path=model_path, use_prompt=True, prompt_doc=prompt_doc, debug=debug), batch_size=batch_size)
    else:
        reranker = Rerank(GPTRanker(model_path=model_path, use_prompt=True, prompt_doc=prompt_doc, debug=debug), batch_size=batch_size)

    for i, dataset in enumerate(datasets):

        logging.info(f"\n{'-' * 10} Running {dataset} {'-' * 10}\n")
        
        if not(os.path.exists(os.path.join(beir_data_dir, dataset))):
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            data_path = util.download_and_unzip(url, beir_data_dir)
            print("Dataset downloaded here: {}".format(data_path))
            
        # Load the dataset into BEIR
        data_path = os.path.join(beir_data_dir, dataset)

        # cqadupstack - Contains several sub datasets
        if dataset == "cqadupstack":
            cqa_ndcgs_bm25, cqa_maps_bm25, cqa_recalls_bm25, cqa_precisions_bm25 = [], [], [], []
            cqa_ndcgs, cqa_maps, cqa_recalls, cqa_precisions = [], [], [], []
            for sub_dataset in os.listdir(data_path):
                sub_data_path = f"datasets/{dataset}/{sub_dataset}"
                
                results_bm25_path = f"results_{dataset}_{sub_dataset}.json"
                results_path = f"results_{model_out_name}_prompt{prompt_id}_{dataset}_{sub_dataset}.json"
                # Skip if already computed these results
                if os.path.exists(os.path.join(os.getcwd(), results_path)):
                    continue

                (ndcg_bm25, _map_bm25, recall_bm25, precision_bm25), (ndcg, _map, recall, precision) = run_reranking(results_bm25_path, results_path, sub_data_path)

                cqa_ndcgs_bm25.append(ndcg)
                cqa_maps_bm25.append(_map)
                cqa_recalls_bm25.append(recall)
                cqa_precisions_bm25.append(precision)

                cqa_ndcgs.append(ndcg)
                cqa_maps.append(_map)
                cqa_recalls.append(recall)
                cqa_precisions.append(precision)

            for (metric, metric_group) in [(ndcg_bm25, cqa_ndcgs_bm25), (_map_bm25, cqa_maps_bm25), (recall_bm25, cqa_recalls_bm25), (precision_bm25, cqa_precisions_bm25)]:
                for k in metric.keys():
                    metric[k] = sum([score[k] for score in metric_group]) / len(metric_group)

            for (metric, metric_group) in [(ndcg, cqa_ndcgs), (_map, cqa_maps), (recall, cqa_recalls), (precision, cqa_precisions)]:
                for k in metric.keys():
                    metric[k] = sum([score[k] for score in metric_group]) / len(metric_group)

            logging.info("CQA Final BM25")
            logging.info(f"{ndcg_bm25}")
            logging.info(f"{_map_bm25}")
            logging.info(f"{recall_bm25}")
            logging.info(f"{precision_bm25}")

            logging.info("CQA Final")
            logging.info(f"{ndcg}")
            logging.info(f"{_map}")
            logging.info(f"{recall}")
            logging.info(f"{precision}")

        else:
            results_bm25_path = f"results_{dataset}.json"
            results_path = f"results_{model_out_name}_prompt{prompt_id}_{dataset}.json"
            # Skip if already computed these results
            if os.path.exists(os.path.join(os.getcwd(), results_path)):
                continue
            (ndcg_bm25, _map_bm25, recall_bm25, precision_bm25), (ndcg, _map, recall, precision) = run_reranking(results_bm25_path, results_path, data_path)

        ndcgs.setdefault(dataset, {})
        ndcgs[dataset]["ndcg"] = ndcg
        ndcgs[dataset]["map"] = _map
        ndcgs[dataset]["recall"] = recall
        ndcgs[dataset]["precision"] = precision

        ndcgs_bm25[dataset] = ndcg_bm25

        # Optionally clean-up each time to avoid running out of space
        # !rm -r datasets

    if os.path.exists(scores_out_path):
        with open(scores_out_path, "r") as f:
            res = json.load(f)
            ndcgs = {**res, **ndcgs}
    with open(scores_out_path, 'w') as fp:
        json.dump(ndcgs, fp)
