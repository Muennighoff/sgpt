from useb import run
from sentence_transformers import SentenceTransformer
import torch

sbert = SentenceTransformer('bert-base-nli-mean-tokens')

@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    return torch.Tensor(sbert.encode(sentences, show_progress_bar=False))

results, results_main_metric = run(
    semb_fn_askubuntu=semb_fn, 
    semb_fn_cqadupstack=semb_fn,  
    semb_fn_twitterpara=semb_fn, 
    semb_fn_scidocs=semb_fn,
    eval_type='test',
    data_eval_path='data-eval'
)

assert round(results_main_metric['avg'], 1) == 47.6