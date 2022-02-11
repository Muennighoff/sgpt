import argparse
import json
import logging
import os
import pathlib
import pickle
from typing import Dict, List, Tuple

import numpy as np
import openai
import pandas as pd
from transformers import GPT2TokenizerFast
from retry import retry

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from dense import DenseRetrievalExactSearch
from parallelizer import DataFrameParallelizer
from parallelizer.parallelizer import ErrorHandling

# Code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logger = logging.getLogger(__name__)

API_KEY = "API_KEY"

# We don't use OpenAIs custom exceptions, as it will raise
# TypeError: catching classes that do not inherit from BaseException is not allowed
API_EXCEPTIONS = (Exception,)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset to embed.")
    parser.add_argument("--engine", type=str, default="ada", help="Engine to use.")
    parser.add_argument("--endpoint", type=str, default="search", help="search / similarity")
    parser.add_argument("--datapath", type=str, default="./", help="Path to folder with datasets")
    parser.add_argument(
        "--overwrite",
        action="store_const",
        default=False,
        const=True,
        help="Whether to recompute & overwrite existing results",
    )
    parser.add_argument("--batchsize", type=int, default=250, help="How many requests to batch")
    parser.add_argument(
        "--parallelworkers",
        type=int,
        default=4,
        help="Num workers sending requests",
    )
    parser.add_argument("--maxattempts", type=int, default=3, help="Maximum number of attempts")
    parser.add_argument(
        "--waitinterval", type=int, default=10, help="Seconds to wait after failed attempt"
    )

    args = parser.parse_args()
    return args


class OpenAIRetriever:
    def __init__(
        self,
        doc_engine="ada-search-document",
        query_engine="ada-search-query",
        endpoint="search",
        api_key=API_KEY,
        dataset="",
        tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"),
        max_query_len=0,
        max_token_len=2048,
        batch_size=250,
        parallel_workers=4,
        max_attempts=3,
        wait_interval=10,
        **kwargs,
    ):

        self.doc_engine = doc_engine
        self.query_engine = query_engine
        self.api_key = api_key

        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_token_len = max_token_len

        if max_query_len >= max_token_len:
            raise ValueError(
                "Longest query exceed maximum tokens - How to rank this with a corpus?"
            )
        engine = doc_engine.split("-")[0]
        base_path = f"embeddings/{endpoint}/{engine}/{dataset}/"
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        self.out_name_base = f"{base_path}/{engine}"

        # Request parameters
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.max_attempts = max_attempts
        self.wait_interval = wait_interval

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        # Embed if not already present
        embedding_queries_path = f"{self.out_name_base}_queries.pickle"
        if os.path.exists(embedding_queries_path):
            embeddings = pickle.load(open(embedding_queries_path, "rb"))
        else:
            embeddings = self.embed(
                texts=queries,
                engine=self.query_engine,
                api_key=self.api_key,
                tokenizer=self.tokenizer,
                max_query_len=self.max_query_len,
                max_token_len=self.max_token_len,
                out_name=embedding_queries_path,
                save_to_file=True,
                batch_size=self.batch_size,
                parallel_workers=self.parallel_workers,
                max_attempts=self.max_attempts,
                wait_interval=self.wait_interval,
            )
        # Sort embeddings according to the order given & take just the values
        embeddings = [embeddings[id] for (id, _) in queries]

        embeddings = np.array(embeddings)
        logging.info(f"Produced embeddings of shape {embeddings.shape}")
        return embeddings

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, batch_num="", **kwargs
    ) -> np.ndarray:
        # Embed if not already present
        embedding_corpus_path = f"{self.out_name_base}_corpus{batch_num}.pickle"
        if os.path.exists(embedding_corpus_path):
            embeddings = pickle.load(open(embedding_corpus_path, "rb"))
        else:
            # corpus is of form [(id, {"title": "xxx", "text": "yyy"}), ...]
            corpus = [(id, data["text"]) for (id, data) in corpus]
            embeddings = self.embed(
                texts=corpus,
                engine=self.doc_engine,
                api_key=self.api_key,
                tokenizer=self.tokenizer,
                max_query_len=self.max_query_len,
                max_token_len=self.max_token_len,
                out_name=embedding_corpus_path,
                save_to_file=True,
                batch_size=self.batch_size,
                parallel_workers=self.parallel_workers,
                max_attempts=self.max_attempts,
                wait_interval=self.wait_interval,
            )
        # Sort embeddings according to the order given
        embeddings = [embeddings[id] for (id, _) in corpus]

        embeddings = np.array(embeddings)
        logging.info(f"Produced embeddings of shape {embeddings.shape}")
        return embeddings

    @staticmethod
    def embed(
        texts: List[Tuple[int, str]],
        engine: str,
        api_key: str,
        tokenizer,
        max_query_len: int,
        max_token_len: int,
        out_name=None,
        save_to_file=False,
        batch_size=250,
        parallel_workers=4,
        max_attempts=3,
        wait_interval=10,
    ):

        openai.api_key = api_key
        logging.info(f"Starting embedding of {len(texts)} texts.")

        df = pd.DataFrame(texts, columns=["id", "txt"])

        @retry(API_EXCEPTIONS, delay=wait_interval, tries=max_attempts)
        def call_gpt_api(
            batch: List[Dict],
            text_column: str = "txt",
            id_column: str = "id",
            decode: bool = True,
        ) -> List[List[float]]:
            """
            Calls GPT API.
            """
            all_tokens = []
            used_indices = []
            for i, row in enumerate(batch):
                txt, id = row[text_column], row[id_column]
                # Recommendation from OpenAI Docs: replace newlines with space
                txt = txt.replace("\n", " ")
                tokens = tokenizer.encode(txt, add_special_tokens=False)
                token_len = len(tokens)
                if token_len == 0:
                    raise ValueError("Empty items should be cleaned prior to running")
                if token_len + max_query_len > max_token_len:
                    tokens = tokens[: max_token_len - max_query_len - 1]  # 0-indexed
                # For some characters the API raises weird errors, e.g. input=[[126]]
                if decode:
                    tokens = tokenizer.decode(tokens)
                all_tokens.append(tokens)
                used_indices.append(i)

            out = [[]] * len(batch)
            if all_tokens:
                response = openai.Engine(id=engine).embeddings(input=all_tokens)

                assert len(response["data"]) == len(
                    all_tokens
                ), f"Sent {len(all_tokens)}, got {len(response['data'])}"

                for data in response["data"]:
                    idx = data["index"]
                    # OpenAI seems to return them ordered, but to be save use the index and insert
                    idx = used_indices[idx]
                    embedding = data["embedding"]
                    out[idx] = embedding
            return out

        df_parallelizer = DataFrameParallelizer(
            function=call_gpt_api,
            error_handling=ErrorHandling.FAIL,
            exceptions_to_catch=API_EXCEPTIONS,
            parallel_workers=parallel_workers,
            output_column_prefix="gpt",
            batch_support=True,
            batch_size=batch_size,
        )

        df = df_parallelizer.run(
            df,
        )

        assert len(df) == len(texts)
        logging.info(f"Embedded {len(df)} texts.")

        def format_results(v):
            # Pandas concats all columns as a list - We only have one column, hence select it
            emb = v[0]
            # Outputted list has been turned into a String so load via json to turn back into list
            emb = json.loads(emb)
            assert isinstance(emb, list), f"Expected list, but got {type(emb)}"
            return emb

        embeddings = df[["id", "gpt_response"]].set_index("id").T.to_dict("list")
        embeddings = {k: format_results(v) for k, v in embeddings.items()}

        if save_to_file:
            pickle.dump(embeddings, open(out_name, "wb"))
        return embeddings


def main(args):
    dataset = args.dataset
    endpoint = args.endpoint
    engine = args.engine
    base_data_path = args.datapath
    overwrite = args.overwrite

    # Request parameters
    batch_size = args.batchsize
    parallel_workers = args.parallelworkers
    max_attempts = args.maxattempts
    wait_interval = args.waitinterval

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

    empty_keys = [k for k, v in corpus.items() if not v["text"]]
    logging.info(f"Found {len(empty_keys)} empty keys in corpus. Removing...")
    assert len(empty_keys) < len(corpus), "Too many empty keys..."
    # Remove keys in place
    for k in empty_keys:
        del corpus[k]

    empty_keys = [k for k, v in queries.items() if not v]
    assert not empty_keys, f"Contains {len(empty_keys)} empty queries"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    max_query_len = max([len(tokenizer.tokenize(q)) for q in queries.values()])

    if endpoint == "search":
        doc_engine = f"{engine}-search-document"
        query_engine = f"{engine}-search-query"
    elif endpoint == "similarity":
        doc_engine = query_engine = f"{engine}-similarity"

    custom_model = DenseRetrievalExactSearch(
        OpenAIRetriever(
            doc_engine=doc_engine,
            query_engine=query_engine,
            endpoint=endpoint,
            dataset=dataset,
            tokenizer=tokenizer,
            max_query_len=max_query_len,
            batch_size=batch_size,
            parallel_workers=parallel_workers,
            max_attempts=max_attempts,
            wait_interval=wait_interval,
        )
    )
    # Turn cqadupstack/english -> cqadupstack_english
    dataset = dataset.replace("/", "_")
    engine = doc_engine.split("-")[0]
    out_path = f"./results_{engine}_{endpoint}_{dataset}.json"
    if os.path.exists(out_path) and not overwrite:
        logging.info(f"Found {out_path} - Skipping ...")
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

    ndgcs_path = f"./beir_openai_{endpoint}_embeddings_ndcgs.json"
    if not os.path.exists(ndgcs_path):
        ndcgs_json = {"ndcgs": {}}
    else:
        with open(ndgcs_path, "r") as f:
            ndcgs_json = json.load(f)

    ndcgs_json["ndcgs"].setdefault(engine, {})
    ndcgs_json["ndcgs"][engine][dataset] = ndcg

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
        f"cqadupstack_{cqadataset}" in ndcgs_json["ndcgs"][engine]
        for cqadataset in CQADUPSTACK_DATASETS
    ):
        ndcgs_json["ndcgs"][engine]["cqadupstack"] = {}
        for cqadataset in CQADUPSTACK_DATASETS:
            for k, v in ndcgs_json["ndcgs"][engine][f"cqadupstack_{cqadataset}"].items():
                ndcgs_json["ndcgs"][engine]["cqadupstack"].setdefault(k, 0)
                ndcgs_json["ndcgs"][engine]["cqadupstack"][k] += v / len(CQADUPSTACK_DATASETS)

    with open(ndgcs_path, "w") as f:
        json.dump(ndcgs_json, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
