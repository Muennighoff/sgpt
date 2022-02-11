### Simple wrappers for using ST models for BEIR - Mostly based on beir.retrieval.models.SentenceBERT ###

from sentence_transformers import SentenceTransformer, models
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np

class SentenceBERTAsym:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
        self.sep = sep
        self.model = SentenceTransformer(model_path, **kwargs)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        queries = [{'QRY': q} for q in queries]
        return self.model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [{'DOCPOS': (doc["title"] + self.sep + doc["text"]).strip()} if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)

class SentenceBERTBOSEOS:
    def __init__(self, 
                model_path: Union[str, Tuple] = None, 
                sep: str = " ",
                speca=False,
                specb=False,
                **kwargs):

        self.sep = sep
        self.model = SentenceTransformer(model_path, **kwargs)

        word_embedding_model = self.model._first_module()
        assert isinstance(word_embedding_model, models.Transformer)
        self.speca = speca
        self.specb = specb

        if self.specb:
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
        
        elif self.speca:
            tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
            word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

            word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("[EOS]", add_special_tokens=False)[0]
            
            word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("{EOS}", add_special_tokens=False)[0]    



    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if self.speca or self.specb:
            # Will be replaced with [ in the models tokenization
            # If we would put [ here, there is a risk of it getting chained with a different token when encoding
            queries = ["[SOS]" + q for q in queries]
        return self.model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if self.speca or self.specb:
            # Will be replaced with { in the models tokenization
            # If we would put { here, there is a risk of it getting chained with a different token when encoding
            sentences = [("{SOS}" + doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else "{SOS}" + doc["text"].strip() for doc in corpus]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)
