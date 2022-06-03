import argparse
import collections
import json
import logging
import os
import pathlib
import pickle
from typing import Dict, List, Tuple, ValuesView

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from custommodels import DenseRetrievalExactSearch, SentenceBERTAsym, SentenceBERTBOSEOS

# Code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logger = logging.getLogger(__name__)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset to embed.")
    parser.add_argument("--modelname", type=str, default="bert-base-uncased", help="Model to use.")
    parser.add_argument("--method", type=str, default="mean", help="Method to use.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
    parser.add_argument("--layeridx", type=int, default=-1, help="Layer to use: -1 is the last.")
    parser.add_argument(
        "--usest",
        action="store_const",
        default=False,
        const=True,
        help="Whether to use Sentence Transformers",
    )
    parser.add_argument("--datapath", type=str, default="./datasets/", help="Path to folder with datasets")
    parser.add_argument(
        "--overwrite",
        action="store_const",
        default=False,
        const=True,
        help="Whether to recompute & overwrite existing results",
    )
    parser.add_argument("--batchsize", type=int, default=250, help="How many requests to batch")
    parser.add_argument(
        "--saveemb",
        action="store_const",
        default=False,
        const=True,
        help="Whether to save embeddings",
    )
    parser.add_argument(
        "--computeavg",
        action="store_const",
        default=False,
        const=True,
        help="Whether to only compute model avgs",
    )
    parser.add_argument(
        "--selectbest",
        action="store_const",
        default=False,
        const=True,
        help="Compute best ckpts",
    )
    parser.add_argument(
        "--speca",
        action="store_const",
        default=False,
        const=True,
        help="Use special token a encoding method",
    )
    parser.add_argument(
        "--specb",
        action="store_const",
        default=False,
        const=True,
        help="Use special brackets encoding method",
    )
    args = parser.parse_args()
    return args

SPECB_QUE_BOS = "["
SPECB_QUE_EOS = "]"

SPECB_DOC_BOS = "{"
SPECB_DOC_EOS = "}"

class CustomEmbedder:
    def __init__(
        self,
        model_name="EleutherAI/gpt-neo-1.3B",
        batch_size=250,
        device="cuda:0",
        save_emb=False,
        reinit=False,
        layeridx=-1,
        method="mean",
        dataset="scifact",
        specb=False,
        **kwargs,
    ):
        self.device = torch.device(device)

        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(self.device)
        if reinit:
            logging.warn("Reiniting all model weights")
            self.model.init_weights()
        self.model.eval()
        self.max_token_len = self.model.config.max_position_embeddings
        # Account for special tokens:
        if "bert" in model_name:
            logging.info("BERT model detected: Reducing token len by 2 to account for [CLS] & [SEP]")
            self.max_token_len -= 2

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # gpt models do not have a padding token by default - Add one and ignore it with the attn mask lateron
        if "gpt" in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size
        self.save_emb = save_emb
        self.layeridx = layeridx
        self.method = method
        
        self.specb = specb
        if specb:
            self.bos_token_q = self.tokenizer.encode(SPECB_QUE_BOS)
            self.eos_token_q = self.tokenizer.encode(SPECB_QUE_EOS)
            self.bos_token_d = self.tokenizer.encode(SPECB_DOC_BOS)
            self.eos_token_d = self.tokenizer.encode(SPECB_DOC_EOS)
        
        self.base_path = f"embeddings/{model_name.split('/')[-1]}/{self.method}/{dataset}"
        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def embed(self, batch, is_query, **kwargs):

        docs_truncated = 0
        toks_truncated = 0
        total_toks = 0

        batch_tokens = collections.defaultdict(list)
        gather_indices = []

        for i, txt in enumerate(batch):
            # Recommendation from OpenAI Docs: replace newlines with space
            txt = txt.replace("\n", " ")

            # Convert string to list of integers according to tokenizer's vocabulary
            tokens = self.tokenizer.tokenize(txt)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            token_len = len(tokens)
            total_toks += token_len
            if token_len > self.max_token_len:
                docs_truncated += 1
                toks_truncated += token_len - self.max_token_len
            elif token_len == 0:
                raise ValueError("Empty items should be cleaned prior to running")
            
            input_dict = self.tokenizer.prepare_for_model(
                tokens[: self.max_token_len], add_special_tokens=True
            )
            if self.specb:
                if is_query:
                    input_dict["input_ids"] = self.bos_token_q + input_dict["input_ids"] + self.eos_token_q
                else:
                    input_dict["input_ids"] = self.bos_token_d + input_dict["input_ids"] + self.eos_token_d
                input_dict["attention_mask"] = [1] + input_dict["attention_mask"] + [1]

            # input_ids: Same as tokens, but with model-specific beginning and end tokens
            # attention_mask: List of 1s for each input_id, i.e. the tokens it should attend to
            batch_tokens["input_ids"].append(input_dict["input_ids"])
            batch_tokens["attention_mask"].append(input_dict["attention_mask"])
            assert len(input_dict["input_ids"]) == len(input_dict["attention_mask"])
            gather_indices.append(len(input_dict["input_ids"]) - 1)  # Account for 0-indexing

        # No need for truncation, as all inputs are now trimmed to less than the models seq length
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        # Move to CPU/GPU
        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
        with torch.no_grad():
            embedded_batch = self.model(**batch_tokens, output_hidden_states=True, **kwargs)

        all_hidden_states = embedded_batch.hidden_states

        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(all_hidden_states[-1].size())
            .float()
        )

        if docs_truncated:
            logging.warn(
                f"Truncated {docs_truncated} out of {len(batch)} documents by {toks_truncated} out of {total_toks}."
            )

        all_hidden_states = [x.cpu() for x in all_hidden_states]

        return all_hidden_states, input_mask_expanded.cpu(), gather_indices, embedded_batch
    
    def embed_batcher(self, texts: List[Tuple[int, str]], is_query, out_name=None, **kwargs):
        all_embeddings = {}
        for i in range(0, len(texts), self.batch_size):
            # Subselect batch_size items
            batch = texts[i : i + self.batch_size]
            ids, sentences = zip(*batch)
            all_hidden_states, input_mask_expanded, gather_indices, embedded_batch = self.embed(sentences, is_query=is_query)
            
            hidden_state = all_hidden_states[self.layeridx]
            if abs(self.layeridx) > len(all_hidden_states):
                raise ValueError(f"Layer Idx {self.layeridx} is larger than the {len(all_hidden_states)} hidden states")

            ### APPLY POOLING ###
            if self.method == "mean":
                # bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                embedding = sum_embeddings / sum_mask
            elif self.method == "meanmean":
                bs, seq_len, hidden_dim = hidden_state.shape
                num_layers = len(all_hidden_states)
                hidden_states = torch.stack(all_hidden_states)

                input_mask_expanded = input_mask_expanded.unsqueeze(0).expand(hidden_states.size())
                assert hidden_states.shape == input_mask_expanded.shape

                # num_layers, bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(
                    torch.sum(hidden_states * input_mask_expanded, dim=2), dim=0
                )
                sum_mask = input_mask_expanded.sum(dim=2).sum(dim=0)

                embedding = sum_embeddings / sum_mask
            elif self.method == "weightedmean":
                weights = (
                    torch.arange(start=1, end=hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .float()
                )
                # bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(hidden_state * input_mask_expanded * weights, dim=1)
                sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

                embedding = sum_embeddings / sum_mask
            elif self.method == "lasttoken":
                bs, seq_len, hidden_dim = hidden_state.shape

                # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
                gather_indices = torch.LongTensor(gather_indices)
                gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
                gather_indices = gather_indices.unsqueeze(1)
                assert gather_indices.shape == (bs, 1, hidden_dim)

                # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
                # No need for the attention mask as we gather the last token where attn_mask = 1
                embedding = torch.gather(hidden_state, 1, gather_indices).squeeze()

            elif self.method == "lasttokenmean":
                bs, seq_len, hidden_dim = hidden_state.shape

                num_layers = len(all_hidden_states)
                hidden_states = torch.stack(all_hidden_states)

                # Turn indices from shape [bs] --> [num_layers, bs, 1, hidden_dim]
                gather_indices = torch.LongTensor(gather_indices)
                gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
                gather_indices = gather_indices.unsqueeze(0).repeat(num_layers, 1, 1)
                gather_indices = gather_indices.unsqueeze(2)
                assert gather_indices.shape == (num_layers, bs, 1, hidden_dim)

                # Gather along the 2nd dim (seq_len) (num_layers, bs, seq_len, hidden_dim -> num_layers, bs, hidden_dim)
                embedding = torch.gather(hidden_states, 2, gather_indices).squeeze()
                assert embedding.shape == (num_layers, bs, hidden_dim)
                # num_layers, bs, hidden_dim -> bs, hidden_dim
                embedding = torch.mean(embedding, 0)

            elif self.method == "poolout":
                embedding = embedded_batch.pooler_output.cpu()

            add_embeddings = {id: emb.numpy() for id, emb in zip(ids, embedding)}
            all_embeddings = {**all_embeddings, **add_embeddings}

        assert len(texts) == len(all_embeddings)

        if self.save_emb:
            pickle.dump(all_embeddings, open(out_name, "wb"))
        
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        # Embed if not already present
        embedding_queries_path = f"{self.base_path}_queries.pickle"
        if os.path.exists(embedding_queries_path):
            embeddings = pickle.load(open(embedding_queries_path, "rb"))
        else:
            embeddings = self.embed_batcher(texts=queries, out_name=embedding_queries_path, is_query=True, **kwargs)

        # Sort embeddings according to the order given & take just the values
        embeddings = [embeddings[id] for (id, _) in queries]

        embeddings = np.array(embeddings)
        logger.info(f"Produced embeddings of shape {embeddings.shape}")
        return embeddings

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, batch_num="", **kwargs
    ) -> np.ndarray:
        # Embed if not already present
        embedding_corpus_path = f"{self.base_path}_corpus{batch_num}.pickle"
        if os.path.exists(embedding_corpus_path):
            embeddings = pickle.load(open(embedding_corpus_path, "rb"))
        else:
            # corpus is of form [(id, {"title": "xxx", "text": "yyy"}), ...]
            corpus = [(id, data["text"]) for (id, data) in corpus]
            embeddings = self.embed_batcher(texts=corpus, out_name=embedding_corpus_path, is_query=False, **kwargs)
        # Sort embeddings according to the order given
        embeddings = [embeddings[id] for (id, _) in corpus]

        embeddings = np.array(embeddings)
        logger.info(f"Produced embeddings of shape {embeddings.shape}")
        return embeddings



def main(args):
    dataset = args.dataset
    model_name = args.modelname
    device = args.device
    use_st = args.usest
    base_data_path = args.datapath
    overwrite = args.overwrite
    batch_size = args.batchsize
    save_emb = args.saveemb
    method = args.method
    layeridx = args.layeridx
    speca = args.speca
    specb = args.specb


    if args.computeavg:
        compute_model_avg()
        exit()
    elif args.selectbest:
        select_best_ckpt()
        exit()

    data_path = f"{base_data_path}/{dataset}"

    if not os.path.exists(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset
        )
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        print("Dataset downloaded here: {}".format(data_path))
        # Load the dataset into BEIR
        data_path = f"datasets/{dataset}"

    # In the paper it says, BEIR used the dev set for msmarco
    split = "dev" if dataset == "msmarco" else "test"

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    corpus = clean_titles(corpus) if "robust04" in data_path else corpus
    empty_keys = [k for k, v in corpus.items() if not v["text"]]
    logger.info(f"Found {len(empty_keys)} empty keys in corpus. Removing...")
    assert len(empty_keys) < len(corpus), "Too many empty keys..."
    # Remove keys in place
    for k in empty_keys:
        del corpus[k]

    empty_keys = [k for k, v in queries.items() if not v]
    assert not empty_keys, f"Contains {len(empty_keys)} empty queries"

    if use_st:
        from beir.retrieval import models
        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
        if "asym" in model_name:
            logger.info(f"Using asymmetric model.")
            custom_model = DRES(SentenceBERTAsym(model_name, device=device), batch_size=batch_size)
        elif speca or specb:
            custom_model = DRES(SentenceBERTBOSEOS(model_name, speca=speca, specb=specb, device=device), batch_size=batch_size)
        else:
            custom_model = DRES(models.SentenceBERT(model_name, device=device), batch_size=batch_size)

    else:
        if speca:
            raise ValueError("speca is only supported with use_st")
        custom_model = DenseRetrievalExactSearch(
            CustomEmbedder(
                model_name=model_name,
                method=method,
                device=device,
                batch_size=batch_size,
                save_emb=save_emb,
                layeridx=layeridx,
                specb=specb,
            )
        )

    # Turn cqadupstack/english -> cqadupstack_english
    dataset = dataset.replace("/", "_")
    model_name = model_name.replace("/", "_")

    out_path = f"./results_{model_name}_{method}_{dataset}.json"
    if os.path.exists(out_path) and not overwrite:
        logger.info(f"Found {out_path} - Skipping ...")
        return
    # Optionally use less k-values to save memory
    # E.g. [.. 100] instead of [.. 1000] will reduce self.results by 90%
    retriever = EvaluateRetrieval(custom_model, k_values=[1, 3, 5, 10, 100, 1000])
    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)
    # Save scores for top 1000 docs for each query, i.e. 1000 * queries lines
    with open(out_path, "w") as fp:
        json.dump(results, fp)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    ndgcs_path = f"./beir_embeddings_ndcgs.json"
    if not os.path.exists(ndgcs_path):
        ndcgs_json = {"ndcgs": {}}
    else:
        with open(ndgcs_path, "r") as f:
            ndcgs_json = json.load(f)

    ndcgs_json["ndcgs"].setdefault(model_name, {})
    ndcgs_json["ndcgs"][model_name][dataset] = ndcg

    # Add average of cqadupstack once all present
    CQADUPSTACK_DATASETS = [
        "android",
        "english",
        "gaming",
        "gis",
        "mathematica",
        "physics",
        "programmers",
        "stats",
        "wordpress",
        "webmasters",
        "unix",
        "tex",
    ]

    if "cqadupstack" in dataset and all(
        f"cqadupstack_{cqadataset}" in ndcgs_json["ndcgs"][model_name]
        for cqadataset in CQADUPSTACK_DATASETS
    ):
        ndcgs_json["ndcgs"][model_name]["cqadupstack"] = {}
        for cqadataset in CQADUPSTACK_DATASETS:
            for k, v in ndcgs_json["ndcgs"][model_name][f"cqadupstack_{cqadataset}"].items():
                ndcgs_json["ndcgs"][model_name]["cqadupstack"].setdefault(k, 0)
                ndcgs_json["ndcgs"][model_name]["cqadupstack"][k] += v / len(CQADUPSTACK_DATASETS)

    with open(ndgcs_path, "w") as f:
        json.dump(ndcgs_json, f)

def clean_titles(corpus):
    for k in corpus:
        if "title" in corpus[k] and corpus[k]["title"] is None:
            corpus[k]["title"] = ""
    return corpus

def compute_model_avg():
    ndgcs_path = f"./beir_embeddings_ndcgs.json"
    if os.path.exists(ndgcs_path):
        with open(ndgcs_path, "r") as f:
            ndcgs_json = json.load(f)

    subsubavg_datasets = ["nfcorpus", "fiqa", "arguana", "scidocs", "scifact"]

    subavg_datasets = ["trec-covid", "nfcorpus", "hotpotqa", "fiqa", "arguana", "webis-touche2020", 
                        "quora", "dbpedia-entity", "fever", "climate-fever", "scifact"]

    # Average does not include msmarco due to in-domain
    avg_datasets = ["nfcorpus", "bioasq", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news", "arguana", "webis-touche2020", "quora", 
                    "dbpedia-entity", "scidocs", "fever", "climate-fever", "scifact", "robust04", "cqadupstack", "trec-covid"]

    for model_name in ndcgs_json["ndcgs"]:
        ndcgs_json["ndcgs"][model_name]["average"] = {}
        ndcgs_json["ndcgs"][model_name]["subaverage"] = {}
        ndcgs_json["ndcgs"][model_name]["subsubaverage"] = {}
        model_datasets = [ds for ds in ndcgs_json["ndcgs"][model_name] if ds in avg_datasets]
        for dataset in ndcgs_json["ndcgs"][model_name]:
            if dataset not in model_datasets:
                print(f"Skipping {dataset}")
                continue
            for k, v in ndcgs_json["ndcgs"][model_name][dataset].items():
                ndcgs_json["ndcgs"][model_name]["average"].setdefault(k, 0)
                ndcgs_json["ndcgs"][model_name]["average"][k] += v / len(model_datasets)
                if all(sub_ds in model_datasets for sub_ds in subavg_datasets) and (dataset in subavg_datasets):
                    ndcgs_json["ndcgs"][model_name]["subaverage"].setdefault(k, 0)
                    ndcgs_json["ndcgs"][model_name]["subaverage"][k] += v / len(subavg_datasets)
                if all(subsub_ds in model_datasets for subsub_ds in subsubavg_datasets) and (dataset in subsubavg_datasets):
                    ndcgs_json["ndcgs"][model_name]["subsubaverage"].setdefault(k, 0)
                    ndcgs_json["ndcgs"][model_name]["subsubaverage"][k] += v / len(subsubavg_datasets)

    with open(ndgcs_path, "w") as f:
        json.dump(ndcgs_json, f)

def select_best_ckpt():
    """A bit hard-coded function for selecting the best checkpoints given results of many ckpts"""
    ndgcs_path = "./beir_embeddings_ndcgs.json"
    if os.path.exists(ndgcs_path):
        with open(ndgcs_path, "r") as f:
            ndcgs_json = json.load(f)
    
    best_ndgcs_path = "./beir_embeddings_best_ndcgs.json"
    if not os.path.exists(best_ndgcs_path):
        best_ndgcs_json = {"ndcgs": {}}
    else:
        with open(best_ndgcs_path, "r") as f:
            best_ndgcs_json = json.load(f)
    
    # SGPT 125M ckpts
    ckpts = ["15600", "31200", "46800", "62398", "62400", "78000",]
    
    # SGPT 2.7B ckpts 
    ckpts += ["101387", "124784", "148181", "156000", "31196", "54593", "7799", "93588",                          
            "109186", "132583", "15598", "38995", "62392", "77990",
            "116985", "140382", "155980", "23397", "46794", "70191", "85789"]

    # SGPT 6.1B ckpts
    ckpts += ["112311",  "137269",  "174706",  "237101",  "262059",  "299496",  "37437",  "74874", 
    "12479",   "149748",  "187185",  "212143",     "24958",   "274538",  "311975",  "49916",  "87353",
    "124790",  "162227",  "199664",  "224622",     "249580",  "287017",  "311990",  "62395",  "99832",]

    ckpts = set(ckpts)

    for model_name in ndcgs_json["ndcgs"]:
        model_ckpt = model_name.split("_")[-1]
        model_base_name = model_name.strip(model_ckpt)
        if model_ckpt in ckpts:
            best_score = 0
            best_model_name = None
            for ckpt in ckpts:
                cur_model_name = model_base_name + ckpt
                if cur_model_name not in ndcgs_json["ndcgs"]:
                    logging.info(f"Did not find {cur_model_name}")
                    continue
                cur_score = ndcgs_json["ndcgs"][cur_model_name]["average"]["NDCG@10"]
                if cur_score > best_score:
                    best_score = cur_score
                    best_model_name = cur_model_name
            best_ndgcs_json["ndcgs"][best_model_name] = ndcgs_json["ndcgs"][best_model_name]
        else:
            logger.info(f"Did not find ckpts for {model_name}. Skipping...")

    with open(best_ndgcs_path, "w") as f:
        json.dump(best_ndgcs_json, f)

def rank_model_avg():
    """A function for quickly ranking the best models - Can just be copy pasted into the local Python Interpreter"""
    import os, json
    ndgcs_path = "./beir_embeddings_best_ndcgs.json"
    if os.path.exists(ndgcs_path):
        with open(ndgcs_path, "r") as f:
            ndcgs_json = json.load(f)

    out = sorted(ndcgs_json["ndcgs"], key=lambda x: ndcgs_json["ndcgs"][x]["average"]["NDCG@10"], reverse=True)
    print({x: ndcgs_json["ndcgs"][x] for x in out[:5]})
    print(out[:5])

if __name__ == "__main__":
    args = parse_args()
    main(args)

