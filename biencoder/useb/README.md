## Bi-Encoder on USEB (Inference)

This module is for evaluating biencoders on the USEB dataset, also see https://github.com/UKPLab/useb.


### Structure

- `useb`: Folder with a slightly adapted version of https://github.com/UKPLab/useb. Refer to the README in the folder for the changes.
- `useb_dense_retriever.py`: Module for benchmarking Bi-Encoder SGPT & SBERT models on USEB.
- `crossencoder_bioasq_bm25`: Parsing of the BioASQ dataset & running it with BM25 - This dataset is 21GB in size & requires some specific processing, hence the separate notebook. Run it on a large RAM instance to avoid BM25 running out of memory.
- `crossencoder_openai`: Scoring the OpenAI semantic search endpoint on BEIR (This is not the embedding endpoint, but (most likely) a Cross Encoder based endpoint)
- `crossencoder_utils`: Various utils for BIER & the other noteboks (Partly they have been duplicated in `crossencoder_beir_sgpt`)


### Downloads

#### Datasets

OOD Unsupervised results: https://www.kaggle.com/muennighoff/useboodsupervisedresults
OOD Unsupervised + OOD Supervised results: https://www.kaggle.com/muennighoff/useboodsupervisedresults

### Commands

Below are the commands used for all models trained in the paper and available at https://huggingface.co/Muennighoff. Before running, make sure to download the datasets following instructions in `sgpt/biencoder/useb/useb/README.md`.

To run & compare all layers:
(Replace *** with any of the bash scripts you are interested in.)

```bash
bash ***.bash
```

To run just one using its last layer as the embedding run e.g.:

```bash
python useb_dense_retriever.py --modelname Muennighoff/SGPT-125M-weightedmean-nli --method weightedmean
```

For the best results in the paper, run either:

```bash
python useb_dense_retriever.py --modelname Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit --method weightedmean
```

Or install sentence-transformers from `sgpt/biencoder/nli_msmarco/sentence-transformers` and run:

```bash
python useb_dense_retriever.py --modelname Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit --usest --method weightedmean
```
