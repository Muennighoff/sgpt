"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import argparse

import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import torch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int) # = eval_batch_size
parser.add_argument("--model_name", required=True)
# Does not make sense to increase the number of negatives (only to increase num of batches based on which to update)
parser.add_argument("--gradient_accumulation", default=1, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--pooling", default="mean", type=str)
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument("--freezenonbias", action="store_true", help="Freeze all except biases in transformer")
parser.add_argument("--addxlinear", default=0, type=int, help="Add x linear layers")
parser.add_argument("--linearthenpool", action="store_true", help="Move linear layers before pooling")
parser.add_argument("--useact", action="store_true", help="Whether to use GELU activation on the Linear layers")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandbwatchlog", default="all", type=str) # Set e.g. to just gradients for large models
parser.add_argument("--learntmean", action="store_true")
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--model_save_path", default=None, type=str)

args = parser.parse_args()
print(args)

model_name = args.model_name #sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'
train_batch_size = args.train_batch_size #64          #The larger you select this, the better the results (usually). But it requires more GPU memory
gradient_accumulation = args.gradient_accumulation
max_seq_length = 75
num_epochs = 1

if "gpt" in model_name:
    accelerator = Accelerator()
else:
    # Needed to run e.g. bert-large-uncased (Can also be used with GPT but will use unnecessary memory)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

if args.wandb and accelerator.is_main_process:
    import wandb
    wandb.init(project="sgpt", entity="muennighoff")
    wandb.config.update(args)

# Save path of the model
if args.model_save_path is None:
    model_save_path = 'output/training_nli_v2_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
else:
    model_save_path = args.model_save_path
    
# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
if args.freeze or args.freezenonbias:
    for name, param in word_embedding_model.named_parameters():
        if args.freezenonbias and "bias" in name:
            # Freeze all except bias
            continue 
        param.requires_grad = False

feats = word_embedding_model.get_word_embedding_dimension()
if args.learntmean:
    pooling_model = models.WeightedMeanPooling(feats, num_positions=max_seq_length)
else:
    pooling_model = models.Pooling(feats, pooling_mode=args.pooling)

linear_modules = []
for i in range(args.addxlinear):
    linear_modules.append(models.Dense(
        in_features=feats, 
        out_features=feats, 
        bias=False if args.freezenonbias else True, # Do not add another bias if we already train all biases (BitFit) 
        activation_function=torch.nn.GELU() if args.useact else torch.nn.Identity(),
        key_name="token_embeddings" if args.linearthenpool else "sentence_embedding",
    ))
if args.linearthenpool:
    modules = [word_embedding_model, *linear_modules, pooling_model]
else:
    modules = [word_embedding_model, pooling_model, *linear_modules]

logging.info(f"Creating SentenceTransformer consisting of {len(modules)} modules")
model = SentenceTransformer(modules=modules)
if "gpt" in model_name:
    model.tokenizer.pad_token = model.tokenizer.eos_token

#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = 'data/AllNLI.tsv.gz'
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if accelerator.is_main_process:
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Read the AllNLI.tsv.gz file and create the training dataset

logging.info("Read AllNLI train dataset")

def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)

train_data = {}
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['label'])
            add_to_samples(sent2, sent1, row['label'])  #Also add the opposite


train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
        train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

logging.info("Train samples: {}".format(len(train_samples)))



# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)


#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

if args.wandb and accelerator.is_main_process:
    wandb.watch(model, log=args.wandbwatchlog, criterion=train_loss, log_freq=100)

# Train the model
if not args.no_training:
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=int(len(train_dataloader)*0.1),
            warmup_steps=warmup_steps,
            optimizer_params={'lr': args.lr},
            gradient_accumulation=gradient_accumulation,
            output_path=model_save_path,
            use_amp=False,          #Set to True, if your GPU supports FP16 operations
            accelerator=accelerator,
            log_wandb=args.wandb
            )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
