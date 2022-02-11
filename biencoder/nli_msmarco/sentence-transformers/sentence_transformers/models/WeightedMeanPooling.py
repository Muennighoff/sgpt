import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class WeightedMeanPooling(nn.Module):
    """
    Token embeddings are weighted mean by their sequence position
    """
    def __init__(self, word_embedding_dimension, num_positions: int = 512, position_start: int = 0, position_weights = None):
        super(WeightedMeanPooling, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'position_start', 'num_positions']
        self.word_embedding_dimension = word_embedding_dimension
        self.position_start = position_start
        self.num_positions = num_positions
        self.position_weights = position_weights if position_weights is not None else nn.Parameter(torch.tensor([1] * (num_positions+1 - position_start), dtype=torch.float))

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # During training token_embeddings may not be padded, so subselect the relevant weights
        positions = token_embeddings.shape[1]
        position_weights = self.position_weights[:positions].unsqueeze(0).unsqueeze(-1).expand(token_embeddings.size())
        assert position_weights.shape == token_embeddings.shape == input_mask_expanded.shape
        input_mask_expanded = input_mask_expanded * position_weights

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        features.update({'sentence_embedding': sum_embeddings / sum_mask})
        return features

    def get_word_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))


    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = WeightedMeanPooling(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
