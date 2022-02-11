### Modified version of BEIRs DenseRetrievalExactSearch to reduce memory footprint & return id-doc pairs ###

import logging
import heapq
import torch
from typing import Dict, List

from beir import util, LoggingHandler
from beir.util import cos_sim, dot_score

# Code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logger = logging.getLogger(__name__)

# Parent class for any dense model
class DenseRetrievalExactSearch:
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {"cos_sim": "Cosine Similarity", "dot": "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: List[int],
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function
                )
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [(qid, queries[qid]) for qid in queries]  # Changed to include the ID
        query_embeddings = self.model.encode_queries(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [(cid, corpus[cid]) for cid in corpus_ids]  # Changed to include the ID

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(
            "Scoring Function: {} ({})".format(
                self.score_function_desc[score_function], score_function
            )
        )

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
                batch_num=batch_num,  ### Only thing I changed
            )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, len(cos_scores[0])),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score

                if batch_num > 0:
                    # Muennighoff:
                    # Reorder by top-K to reduce size of self.results saving memory
                    # This is O(n log (k) and reduces self.results size to 1/(num_batches)
                    k_keys_sorted_by_values = heapq.nlargest(
                        min(top_k + 1, len(self.results[query_id])),
                        self.results[query_id],
                        key=self.results[query_id].get,
                    )
                    self.results[query_id] = {
                        k: self.results[query_id][k] for k in k_keys_sorted_by_values
                    }

        return self.results
