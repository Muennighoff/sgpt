"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
from datetime import datetime

import numpy as np
import torch.cuda
import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--steps_per_epoch", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None,
                    help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--train_dataset_max_size", default=None, type=int)
parser.add_argument("--dev_corpus_max_size", default=-1, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--model_save_path", default=None, type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--add_special_token", action="store_true", help="Special tokens used by OpenAI with lasttoken pooling")
parser.add_argument("--speca", action="store_true")
parser.add_argument("--specb", action="store_true")
parser.add_argument("--asym", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandbwatchlog", default="all", type=str) # Set e.g. to just gradients for large models
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument("--freezenonbias", action="store_true", help="Freeze all except biases in transformer")
parser.add_argument("--unfreezewte", action="store_true", help="Unfreeze Word Token Embeddings")
parser.add_argument("--gradcache", action="store_true")
parser.add_argument("--chunksize",  default=1, type=int, help="Chunks to use for gradcache")
args = parser.parse_args()
print(args)

        
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = args.train_batch_size  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train


if "gpt" in model_name:
    accelerator = Accelerator()
else:
    # Needed to run e.g. bert-large-uncased (Can also be used with GPT but will use unnecessary memory)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

if args.wandb and accelerator.is_main_process:
    import wandb
    wandb.init(project="sgpt", entity="")
    wandb.config.update(args)

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
elif args.asym:
    logging.info("Create new asymmetric SBERT model")
    w1 = models.Transformer(model_name, max_seq_length=max_seq_length)
    w2 = models.Transformer(model_name, max_seq_length=max_seq_length)
    if args.add_special_token or args.speca:
        if args.add_special_token:
            tokens = ["[DOC]", "[QRY]"]
        elif args.speca:
            tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
            w1.bos_spec_token = w1.tokenizer.encode("[SOS]", add_special_tokens=False)
            w1.eos_spec_token = w1.tokenizer.encode("[EOS]", add_special_tokens=False)
            w2.bos_spec_token = w2.tokenizer.encode("[SOS]", add_special_tokens=False)
            w2.eos_spec_token = w2.tokenizer.encode("[EOS]", add_special_tokens=False)
        w1.tokenizer.add_tokens(tokens, special_tokens=True)
        w2.tokenizer.add_tokens(tokens, special_tokens=True)
        w1.auto_model.resize_token_embeddings(len(w1.tokenizer))
        w2.auto_model.resize_token_embeddings(len(w2.tokenizer))
    if "gpt" in model_name:
        w1.tokenizer.pad_token = w1.tokenizer.eos_token
        w2.tokenizer.pad_token = w2.tokenizer.eos_token
    assert w1.get_word_embedding_dimension() == w2.get_word_embedding_dimension()
    # Pooling has no weights, hence can be shared
    pooling = models.Pooling(w1.get_word_embedding_dimension(), args.pooling)

    asym_model = models.Asym({'QRY': [w1], 'DOCPOS': [w2], 'DOCNEG': [w2]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[asym_model, pooling])
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    if "gpt" in model_name:
        word_embedding_model.tokenizer.pad_token = word_embedding_model.tokenizer.eos_token
    
    if args.add_special_token or args.speca:
        if args.add_special_token:
            tokens = ["[DOC]", "[QRY]"]
        elif args.speca:
            tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        if args.speca:
            word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("[EOS]", add_special_tokens=False)[0]
            
            word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("{EOS}", add_special_tokens=False)[0]      
        
    elif args.specb:
        tokens = ["[SOS]", "{SOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]

        word_embedding_model.bos_spec_token_q_rep = word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]
        
        word_embedding_model.bos_spec_token_d_rep = word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]

        word_embedding_model.replace_bos = True

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

if args.freeze or args.freezenonbias:
    for name, param in model.named_parameters():
        if args.freezenonbias and "bias" in name:
            # Freeze all except bias
            continue 
        if args.unfreezewte and "wte" in name:
            # Do not freeze Word Token Embeddings
            continue
        param.requires_grad = False


if args.model_save_path is None:
    model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"),
                                                                                ce_score_margin,
                                                                                datetime.now().strftime(
                                                                                    "%Y-%m-%d_%H-%M-%S"))
else:
    model_save_path = args.model_save_path

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

if not args.no_training:
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages

    if accelerator.is_main_process:
        # Downloads
        #### Read the corpus files, that contain all the passages. Store them in the corpus dict
        if not os.path.exists(collection_filepath):
            tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
            if not os.path.exists(tar_filepath):
                logging.info("Download collection.tar.gz")
                util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)

        if not os.path.exists(queries_filepath):
            tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
            if not os.path.exists(tar_filepath):
                logging.info("Download queries.tar.gz")
                util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)

        # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
        # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
        if not os.path.exists(ce_scores_file):
            logging.info("Download cross-encoder scores file")
            util.http_get(
                'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
                ce_scores_file)

        # As training data we use hard-negatives that have been mined using various systems
        if not os.path.exists(hard_negatives_filepath):
            logging.info("Download cross-encoder scores file")
            util.http_get(
                'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz',
                hard_negatives_filepath)

    ### Read the train queries, store in queries dict
    queries = {}  # dict in the format: query_id -> query. Stores all training queries

    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for idx, line in enumerate(fIn):
            qid, query = line.strip().split("\t")
            qid = int(qid)
            if args.add_special_token:
                query = "[QRY]" + query
            elif args.speca or args.specb:
                query = "[SOS]" + query
            if idx == 0:
                logging.info(f"Train Query Example: {query}")
            queries[qid] = query

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for idx, line in enumerate(fIn):
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            if args.add_special_token:
                passage = "[DOC]" + passage
            elif args.speca or args.specb:
                passage = "{SOS}" + passage
            if idx == 0:
                logging.info(f"Train Doc Example: {passage}")
            corpus[pid] = passage

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    with gzip.open(hard_negatives_filepath, 'rt') as fIn:
        for i, line in tqdm.tqdm(enumerate(fIn)):
            data = json.loads(line)

            # Get the positive passage ids
            qid = data['qid']
            pos_pids = data['pos']

            if len(pos_pids) == 0:  # Skip entries without positives passages
                continue

            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            # Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                if args.negs_to_use is not None:  # Use specific system for negatives
                    negs_to_use = args.negs_to_use.split(",")
                else:  # Use all systems
                    negs_to_use = list(data['neg'].keys())
                logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

            for system_name in negs_to_use:
                if system_name not in data['neg']:
                    continue

                system_negs = data['neg'][system_name]
                negs_added = 0
                for pid in system_negs:
                    if ce_scores[qid][pid] > ce_score_threshold:
                        continue

                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids,
                                              'neg': neg_pids}

            if args.train_dataset_max_size is not None and i > args.train_dataset_max_size:
                break

    logging.info("Train queries: {}".format(len(train_queries)))


    # We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
    # on-the-fly based on the information from the mined-hard-negatives jsonl file.
    class MSMARCODataset(Dataset):
        def __init__(self, queries, corpus, asym=False):
            self.queries = queries
            self.queries_ids = list(queries.keys())
            self.corpus = corpus

            self.asym = asym

            for qid in self.queries:
                self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
                self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
                random.shuffle(self.queries[qid]['neg'])

        def __getitem__(self, item):
            query = self.queries[self.queries_ids[item]]
            query_text = query['query']

            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)

            neg_id = query['neg'].pop(0)  # Pop negative and add at end
            neg_text = self.corpus[neg_id]
            query['neg'].append(neg_id)

            if self.asym:
                return InputExample(texts=[{'QRY': query_text}, {'DOCPOS': pos_text}, {'DOCNEG': neg_text}])

            return InputExample(texts=[query_text, pos_text, neg_text])

        def __len__(self):
            return len(self.queries)

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = MSMARCODataset(train_queries, corpus=corpus, asym=args.asym)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    if args.gradcache:
        train_loss = losses.MNRLGradCache(model, chunk_size=args.chunksize)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)

    if args.wandb and accelerator.is_main_process:
        wandb.watch(model, log=args.wandbwatchlog, criterion=train_loss, log_freq=100)
    
    # Always take 1 ckpt per epoch - If 1 device -> 1 ckpt after whole trainloader
    # If X devices; Train-loader is X times bigger than actual steps
    checkpoint_save_steps = len(train_dataloader) // accelerator.num_processes
    logging.info(f"Dataloader length: {len(train_dataloader)}, CKPT Save Steps: {checkpoint_save_steps}")

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=args.warmup_steps,
              use_amp=args.use_amp,
              checkpoint_path=model_save_path,
              checkpoint_save_steps=checkpoint_save_steps,
              optimizer_params={'lr': args.lr},
              show_progress_bar=True,
              steps_per_epoch=args.steps_per_epoch,
              accelerator=accelerator,
              log_wandb=args.wandb,
              use_gradcache=args.gradcache,
              chunk_size=args.chunksize,
              )

    # Save the model
    model.save(model_save_path)

# Evaluate
### Load eval data
collection_filepath = os.path.join(data_folder, 'collection.tsv')
dev_queries_file = os.path.join(data_folder, 'queries.dev.small.tsv')
qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')

dev_corpus = {}  # Our corpus pid => passage
dev_queries = {}  # Our dev queries. qid => query
dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

### Download files if needed
if accelerator.is_main_process:
    if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
        tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download: " + tar_filepath)
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz',
                          tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    if not os.path.exists(qrels_filepath):
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)

# Load the 6980 dev queries
with open(dev_queries_file, encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        query = query.strip()
        if args.add_special_token:
            query = "[QRY]" + query
        elif args.speca or args.specb:
            query = "[SOS]" + query
        if args.asym:
            query = {'QRY': query}
        dev_queries[qid] = query

# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)

# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage

        if pid in needed_pids or args.dev_corpus_max_size <= 0 or len(dev_corpus) <= args.dev_corpus_max_size:
            passage = passage.strip()
            if args.add_special_token:
                passage = "[DOC]" + passage
            elif args.speca or args.specb:
                passage = "{SOS}" + passage
            if args.asym:
                # The encoder for DOCPOS & DOCNEG is the same, so we can use either one as the key
                passage = {'DOCPOS': passage}
            dev_corpus[pid] = passage

model = SentenceTransformer(model_save_path)

if args.add_special_token or args.speca:

    word_embedding_model = model._first_module()
    assert isinstance(word_embedding_model, models.Transformer)

    if args.add_special_token:
        tokens = ["[DOC]", "[QRY]"]
    elif args.speca:
        tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    if args.speca:
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("[EOS]", add_special_tokens=False)[0]
        
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("{EOS}", add_special_tokens=False)[0]      
    
elif args.specb:

    word_embedding_model = model._first_module()
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



# only performing evaluation from one process
ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs,
                                                        show_progress_bar=True,
                                                        corpus_chunk_size=100000,
                                                        precision_recall_at_k=[10],
                                                        batch_size=args.eval_batch_size,
                                                        name="msmarco dev")
ir_evaluator(model)
