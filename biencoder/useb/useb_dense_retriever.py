import argparse
import collections
import logging
import os
import pathlib
import pickle
import random

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer, GPT2TokenizerFast

from useb.useb import run


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="mean", help="Method to use.")
    parser.add_argument("--modelname", type=str, default="bert-base-uncased", help="Model to use.")
    # Won't make a difference for inference, as inference is deterministic
    parser.add_argument("--seed", type=int, default=42, help="Seed to use.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
    parser.add_argument("--layeridx", type=int, default=-1, help="Layer to use: -1 is the last.")
    parser.add_argument(
        "--notnormalize",
        action="store_const",
        default=False,
        const=True,
        help="Whether not to normalize",
    )
    parser.add_argument(
        "--reinit",
        action="store_const",
        default=False,
        const=True,
        help="Whether to reinit weights of the model (I.e. evaluate with a random model)",
    )
    parser.add_argument(
        "--usest",
        action="store_const",
        default=False,
        const=True,
        help="Whether to use sentence-transformers",
    )
    parser.add_argument(
        "--openai",
        action="store_const",
        default=False,
        const=True,
        help="Use OpenAI's embedding API - Make sure to modify the API_KEY variable",
    )
    parser.add_argument(
        "--saveemb",
        action="store_const",
        default=False,
        const=True,
        help="Whether to save embeddings",
    )
    args = parser.parse_args()
    return args

def set_all_seeds(seed):
    """
    Seed function - Not used here, as inference on GPT models is deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomEmbedder:
    def __init__(
        self,
        model_name="EleutherAI/gpt-neo-1.3B",
        batch_size=250,
        device="cuda:0",
        save_emb=False,
        reinit=False,
        layeridx=-1,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(self.device)
        if reinit:
            logging.warn("Reiniting all model weights")
            self.model.init_weights()
        self.model.eval()
        self.max_token_len = self.model.config.max_position_embeddings
        # Account for special tokens:
        if "bert" in model_name:
            logging.info(
                "BERT model detected: Reducing token len by 2 to account for [CLS] & [SEP]"
            )
            self.max_token_len -= 2

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # gpt models do not have a padding token by default - Add one and ignore it with the attn mask lateron
        if "gpt" in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size
        self.save_emb = save_emb
        self.layeridx = layeridx

        self.base_path = f"embeddings/{model_name.split('/')[-1]}/"
        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def embed(self, batch, **kwargs):

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

    def encode(
        self,
        sentences,
        method="mean",
        show_progress_bar=False,
        dataset_name=None,
        add_name="",
        idx=None,
        **kwargs,
    ):

        out = []
        for i in range(0, len(sentences), self.batch_size):

            ### GET EMBEDDING ###
            embedding_path = f"{self.base_path}/{dataset_name}_{add_name}_{idx}_{i}.pickle"
            if dataset_name is not None and idx is not None and os.path.exists(embedding_path):
                loaded = pickle.load(open(embedding_path, "rb"))
                all_hidden_states = loaded["all_hidden_states"]
                input_mask_expanded = loaded["input_mask_expanded"]
                gather_indices = loaded["gather_indices"]
                hidden_state = all_hidden_states[self.layeridx]
            else:
                # Subselect batch_size items
                batch = sentences[i : i + self.batch_size]
                all_hidden_states, input_mask_expanded, gather_indices, embedded_batch = self.embed(
                    batch, **kwargs
                )
                hidden_state = all_hidden_states[self.layeridx]

                # Save embeddings
                if dataset_name is not None and idx is not None and self.save_emb:
                    dump = {
                        "all_hidden_states": all_hidden_states,
                        "input_mask_expanded": input_mask_expanded,
                        "gather_indices": gather_indices,
                    }
                    pickle.dump(dump, open(embedding_path, "wb"))

            if abs(self.layeridx) > len(all_hidden_states):
                raise ValueError(
                    f"Layer Idx {self.layeridx} is larger than the {len(all_hidden_states)} hidden states"
                )

            ### APPLY POOLING ###
            if method == "mean":
                # bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                embedding = sum_embeddings / sum_mask
            elif method == "meanmean":
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
            elif method == "weightedmean":
                weights = (
                    torch.arange(start=1, end=hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .float()
                )
                assert weights.shape == hidden_state.shape == input_mask_expanded.shape
                # bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(hidden_state * input_mask_expanded * weights, dim=1)
                sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

                embedding = sum_embeddings / sum_mask
            elif method == "learntmean":
                # We assume that learnt mean has weights located at 1_WeightedMeanPooling
                # a bit hardcoded & weights could be fed in via a path
                weights = torch.load(f"{self.model_name}/1_WeightedMeanPooling/pytorch_model.bin")
                positions = hidden_state.shape[1]
                weights = (
                    weights["position_weights"][:positions]
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .to(hidden_state.device)
                )
                assert weights.shape == hidden_state.shape == input_mask_expanded.shape
                # bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(hidden_state * input_mask_expanded * weights, dim=1)
                sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

                embedding = sum_embeddings / sum_mask
            elif method == "lasttoken":
                bs, seq_len, hidden_dim = hidden_state.shape

                # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
                gather_indices = torch.LongTensor(gather_indices)
                gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
                gather_indices = gather_indices.unsqueeze(1)
                assert gather_indices.shape == (bs, 1, hidden_dim)

                # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
                # No need for the attention mask as we gather the last token where attn_mask = 1
                embedding = torch.gather(hidden_state, 1, gather_indices).squeeze()
            elif method == "lasttokenmean":
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
            elif method == "poolout":
                embedding = embedded_batch.pooler_output.cpu()

            # Turn into list
            out.extend(embedding.numpy().tolist())

        assert len(sentences) == len(out)

        return out


class STGPTWrapper:
    """
    Results are identical to using GPTEmbedder with mean pooling.
    """

    def __init__(self, model_name_or_path, max_seq_length=None, **kwargs):
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

API_KEY = "YOUR_KEY"

class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint on USEB.
    """
    def __init__(self, engine, batch_size=250, save_emb=False, **kwargs):
        

        self.engine = engine
        self.max_token_len = 2048
        self.batch_size = batch_size
        self.save_emb = save_emb
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(self, 
            sentences,
            decode=True,                
            method="lasttoken",
            show_progress_bar=False,
            dataset_name=None,
            add_name="",
            idx=None,
            **kwargs):

        import openai
        openai.api_key = API_KEY

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{dataset_name}_{add_name}_{idx}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                all_tokens = []
                used_indices = []
                for j, txt in enumerate(batch):
                    # Recommendation from OpenAI Docs: replace newlines with space
                    txt = txt.replace("\n", " ")
                    tokens = self.tokenizer.encode(txt, add_special_tokens=False)
                    token_len = len(tokens)
                    if token_len == 0:
                        raise ValueError("Empty items should be cleaned prior to running")
                    if token_len > self.max_token_len:
                        tokens = tokens[:self.max_token_len]
                    # For some characters the API raises weird errors, e.g. input=[[126]]
                    if decode:
                        tokens = self.tokenizer.decode(tokens)
                    all_tokens.append(tokens)
                    used_indices.append(j)

                out = [[]] * len(batch)
                if all_tokens:
                    response = openai.Engine(id=self.engine).embeddings(input=all_tokens)
                    assert len(response["data"]) == len(
                        all_tokens
                    ), f"Sent {len(all_tokens)}, got {len(response['data'])}"

                    for data in response["data"]:
                        idx = data["index"]
                        # OpenAI seems to return them ordered, but to be save use the index and insert
                        idx = used_indices[idx]
                        embedding = data["embedding"]
                        out[idx] = embedding
                        
                fin_embeddings.extend(out)
        # Save embeddings
        if fin_embeddings and self.save_emb:
            embedding_path
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings




def main(args):
    method = args.method
    model_name = args.modelname
    device = args.device
    layeridx = args.layeridx
    notnormalize = args.notnormalize
    normalize = not (notnormalize)
    reinit = args.reinit
    save_emb = args.saveemb

    ### MODEL HANDLING ###
    MODELNAME_TO_MODEL = {
        # Zero-shot
        "glove": lambda: SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        ),
        # Out-of-domain supervised
        # To avoid having to add compatibility for linear layers in this Wrapper, 
        # just use Sentence Transformers (--usest)
        # IMPORTANT: Use the sentence-transformers from this repository (biencoder/nli_msmarco/sentence-transformers)
        # as it has been adjusted to work with Dense layers prior to pooling
        # With the normal sentence-transformers there will be a key error when trying to load the Dense layers
        # Just install it using the setup.py in biencoder/nli_msmarco/sentence-transformers
        "SGPT-125M-learntmean-nli": lambda: STGPTWrapper(
            "SGPT-125M-learntmean-nli", max_seq_length=75, device=device
        ),
    }
    if model_name in MODELNAME_TO_MODEL:
        model = MODELNAME_TO_MODEL[model_name]()
    elif args.usest:
        model = STGPTWrapper(model_name, device=device)
    elif args.openai:
        model = OpenAIEmbedder(engine=model_name, save_emb=save_emb)
    else:
        model = CustomEmbedder(
            model_name,
            device=device,
            layeridx=layeridx,
            reinit=reinit,
            save_emb=save_emb
        )

    ### FN HANDLING ###
    @torch.no_grad()
    def semb_mean_fn(sentences, **kwargs) -> torch.Tensor:
        return torch.Tensor(model.encode(sentences, show_progress_bar=False))

    @torch.no_grad()
    def semb_lasttoken_fn(sentences, dataset_name=None, add_name="", idx=None) -> torch.Tensor:
        return torch.Tensor(
            model.encode(
                sentences,
                method="lasttoken",
                show_progress_bar=False,
                dataset_name=dataset_name,
                add_name=add_name,
                idx=idx,
            )
        )

    @torch.no_grad()
    def semb_lasttokenmean_fn(sentences, dataset_name=None, add_name="", idx=None) -> torch.Tensor:
        return torch.Tensor(
            model.encode(
                sentences,
                method="lasttokenmean",
                show_progress_bar=False,
                output_hidden_states=True,
                dataset_name=dataset_name,
                add_name=add_name,
                idx=idx,
            )
        )

    @torch.no_grad()
    def semb_weightedmean_fn(sentences, dataset_name=None, add_name="", idx=None) -> torch.Tensor:
        return torch.Tensor(
            model.encode(
                sentences,
                method="weightedmean",
                show_progress_bar=False,
                dataset_name=dataset_name,
                add_name=add_name,
                idx=idx,
            )
        )

    @torch.no_grad()
    def semb_meanmean_fn(sentences, dataset_name=None, add_name="", idx=None) -> torch.Tensor:
        return torch.Tensor(
            model.encode(
                sentences,
                method="meanmean",
                show_progress_bar=False,
                dataset_name=dataset_name,
                add_name=add_name,
                idx=idx,
            )
        )

    @torch.no_grad()
    def semb_poolout_fn(sentences, dataset_name=None, add_name="", idx=None) -> torch.Tensor:
        return torch.Tensor(
            model.encode(
                sentences,
                method="poolout",
                show_progress_bar=False,
                dataset_name=dataset_name,
                add_name=add_name,
                idx=idx,
            )
        )

    def semb_random_base(sentences, **kwargs) -> torch.Tensor:
        embedding_dim = 8
        return torch.Tensor(np.random.rand(len(sentences), embedding_dim))

    METHOD_TO_FN = {
        "mean": semb_mean_fn,
        "lasttoken": semb_lasttoken_fn,
        "lasttokenmean": semb_lasttokenmean_fn,
        "weightedmean": semb_weightedmean_fn,
        "meanmean": semb_meanmean_fn,
        "poolout": semb_poolout_fn,
        "random": semb_random_base,
    }
    # If a sentence-transformer model, pooling will be automatically loaded & applied
    if args.usest:
        semb_fn = METHOD_TO_FN["mean"]
    # OpenAI uses lasttoken pooling
    # We don't do pooling but using this fn instead of mean gives us kwargs for saving
    elif args.openai:
        semb_fn = METHOD_TO_FN["lasttoken"]
    else:
        semb_fn = METHOD_TO_FN[method]

    ### RUNNING ###
    results, results_main_metric = run(
        semb_fn_askubuntu=semb_fn,
        semb_fn_cqadupstack=semb_fn,
        semb_fn_twitterpara=semb_fn,
        semb_fn_scidocs=semb_fn,
        eval_type="test",
        data_eval_path="data-eval",  # This should be the path to the folder of data-eval
        normalize=normalize,
    )

    # Rename the json files run creates
    model_name = model_name.replace("/", "_")
    method = f"reinit{method}" if reinit else method
    os.rename(
        "results.detailed.json", f"{model_name}_{method}_layer{layeridx}_results_detailed.json"
    )
    os.rename(
        "results.average_precision.json",
        f"{model_name}_{method}_layer{layeridx}_results_average_precision.json",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
