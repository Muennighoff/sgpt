## Cross-Encoder on BEIR (Inference)

### Structure

Notebook overview:

- `crossencoder_beir_bm25.ipynb`: Creating BM25 results on BEIR 
- `crossencoder_beir_sgpt.ipynb`: Creating SGPT reranking results based on BM25 results (Note: If you do not want to rerun BM25, you can download the BM25 results from the datasets provided, see the Downloads section for more information.)
- `crossencoder_bioasq_bm25.ipynb`: Parsing of the BioASQ dataset & running it with BM25 - This dataset is 21GB in size & requires some specific processing, hence the separate notebook. Run it on a large RAM instance to avoid BM25 running out of memory.
- `crossencoder_openai.ipynb`: Scoring the OpenAI semantic search endpoint on BEIR (This is not the embedding endpoint, but (most likely) a Cross Encoder based endpoint)
- `openai_search_endpoint_functionality.py` OpenAI Search Endpoint mechanism (Released to the public on 06/2022 (After SGPT paper release))
- `sgptce.py` Provides a non-notebook variant of `crossencoder_beir_sgpt.ipynb`
- `scripts` Slurm scripts for `sgptce.py`
- `../../other/sgpt_utils`: Various utils for compouting re-ranking scores & graphs for the SGPT paper (The code is partly duplicated in `crossencoder_beir_sgpt`)

### Downloads

#### Datasets

BEIR dataset: https://github.com/UKPLab/beir.

BEIR BM25 results: https://www.kaggle.com/muennighoff/beirbm25results
You can use these BM25 results to benchmark your own re-ranker on BEIR without having to re-run BM25 + It will be easily comparable to all results in the SGPT paper :)
BEIR OpenAI Semantic Search results: https://www.kaggle.com/muennighoff/beiropenairesults
BEIR SGPT Cross-Encoder using GPT-J (6.1B parameters) results: https://www.kaggle.com/muennighoff/beirgptjresults
BEIR SGPT Cross-Encoder using GPT-Neo (2.7B parameters) results: https://www.kaggle.com/muennighoff/beirgptneo27results
BEIR SGPT Cross-Encoder using GPT-Neo (1.3B parameters) results: https://www.kaggle.com/muennighoff/beirgptneo13results
BEIR SGPT Cross-Encoder using GPT-Neo (125M parameters) results: https://www.kaggle.com/muennighoff/beirgptneo01results
BEIR SGPT Cross-Encoder using a 13B model results (not better than 125M): https://www.kaggle.com/muennighoff/beir13bresults
BEIR-Subset prompt ablations: https://www.kaggle.com/muennighoff/beirpromptablations

BEIR Quora Symmetric Semantic Search Experiments Results https://www.kaggle.com/muennighoff/beirquorapromptsresults


### Commands

Just run selected cells in the individual notebooks. Don't hesitate to open an issue should you have any questions~
