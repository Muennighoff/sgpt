## Bi-Encoder on BEIR (Inference)

This module is for evaluating biencoders on the BEIR dataset.

### Structure

- `custommodels`: Provides BEIR compatibility for asymmetric mdoels & models with special tokens
- `io_utils`, `parallelizer`, `beir_openai_embeddings_batched_parallel.py`: Tools for querying the OpenAI embedding endpoint parallelized & in batches
- `beir_dense_retriever.py`: Module for benchmarking Bi-Encoder SGPT & SBERT models on BEIR.
- `requirements.txt`: Contains requirements specific for using `io_utils` & `parallelizer`. For all other the requirements in `sgpt/requirements.txt` with the sentence-transformers module from `sgpt/biencoder/nli_msmarco/sentence-transformers/` should suffice.

### Downloads

#### Datasets

BEIR dataset: https://github.com/UKPLab/beir.

#### Asymmetric Bi-Encoder results

- https://www.kaggle.com/muennighoff/sgptbeasym (Does not include some json score files for 5.8B due to their large size (BioASQ, MSMARCO..))

#### Symmetric Bi-Encoder results (mostly Quora)

- https://www.kaggle.com/muennighoff/beirbiencoderresults1
- https://www.kaggle.com/muennighoff/beirbiencoderresults2
- https://www.kaggle.com/muennighoff/beirbiencoderresults3

#### Models

Find them at https://huggingface.co/Muennighoff.

### Commands

Below are the commands used for all models trained in the paper and available at https://huggingface.co/Muennighoff.

#### SGPT Models

For the best results in the paper on all of BEIR, run:

```bash
bash run_sgpt.bash Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit cuda:0
```

Individual datasets can be run with e.g.:

```bash
python beir_dense_retriever.py --modelname Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit --method weightedmean --dataset scifact --specb --maxseqlen 300
```

Results are accumulated in a .json file. 
To compute the average:

```bash
python beir_dense_retriever.py --computeavg
```

To extract the best checkpoints (not relevant if not training):

```bash
python beir_dense_retriever.py --selectbest
```

To rank models after extraction, you can use the code in the `rank_model_avg()` function.


#### OpenAI Models

Edit `beir_openai_embeddings_batched_parallel.py` to add your API_KEY here `API_KEY = "API_KEY"`.
Make sure to have all BEIR datasets in a folder in `../datasets/`.
The below will cost a lot of money, so proceed with caution.
Run:

```bash
bash run_ada_search.bash
```

```bash
bash run_ada_similarity.bash
```

```bash
bash run_curie_search.bash
```

```bash
bash run_curie_similarity.bash
```


#### Quick benchmarking

You can use the below script for quick benchmarking after you have installed the requirements outlined at the top.

```python
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


from sentence_transformers import SentenceTransformer, models

from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np

class SentenceBERTBOSEOS:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
        self.sep = sep
        self.model = SentenceTransformer(model_path, **kwargs)

        word_embedding_model = self.model._first_module()
        assert isinstance(word_embedding_model, models.Transformer)

        tokens = ["[SOS]", "{SOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        # Will be replaced with the rep ones
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]

        word_embedding_model.bos_spec_token_q_rep = word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]
        
        word_embedding_model.bos_spec_token_d_rep = word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]

        word_embedding_model.replace_bos = True

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # Will be replaced with [ in the models tokenization
        # If we would put [ here, there is a risk of it getting chained with a different token when encoding
        queries = ["[SOS]" + q for q in queries]
        return self.model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # Will be replaced with { in the models tokenization
        # If we would put { here, there is a risk of it getting chained with a different token when encoding
        sentences = [("{SOS}" + doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else "{SOS}" + doc["text"].strip() for doc in corpus]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)

 
#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path("./").parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the SBERT model and retrieve using cosine-similarity

model = DRES(
    SentenceBERTBOSEOS(
        "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    ), batch_size=16
)

retriever = EvaluateRetrieval(model, score_function="cos_sim")
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```
