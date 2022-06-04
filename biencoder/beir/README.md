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
python beir_dense_retriever.py --modelname Muennighoff/SGPT-125M-weightedmean-msmarco --method weightedmean --dataset scifact --specb
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
