## Bi-Encoder on USEB (Inference)

This module is for evaluating biencoders on the USEB dataset, also see https://github.com/UKPLab/useb.


### Structure

- `useb`: Folder with a slightly adapted version of https://github.com/UKPLab/useb. Refer to the README in the folder for the changes.
- `useb_dense_retriever.py`: Module for benchmarking Bi-Encoder SGPT & SBERT models on USEB.


### Downloads

#### Datasets

OOD Unsupervised results: https://www.kaggle.com/muennighoff/useboodsupervisedresults
OOD Unsupervised + OOD Supervised results: https://www.kaggle.com/muennighoff/useboodsupervisedresults

### Commands

#### Requirements

Pip:
```bash
python -m useb/useb.downloading all # Or: python useb/useb/downloading.py all
pip install pytrec_eval
cd ../nli_msmarco/sentence-transformers; pip install -e .
```

Conda:
```bash
python -m useb/useb.downloading all # Or: python useb/useb/downloading.py all
~/conda/envs/sgpt/bin/pip install pytrec_eval
cd ../nli_msmarco/sentence-transformers; ~/conda/envs/sgpt/bin/pip install -e . 
```

#### USEB Inference

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

AA Transformer:

```bash
python useb_dense_retriever.py --modelname /mnt/train-fss/text_search_sym_13B_noprojbiases/global_step13000/ --aa --method nopool --tokenizername /mnt/train-fss/data/128k-alpha-001-by-source/alpha-001-128k.json
```

