from dataclasses import dataclass
from typing import Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_path: str = "./data/examples.parquet"
    split: Optional[str] = "train"
    subset: Optional[str] = None
    text_key: Optional[str] = "text"
    image_key: Optional[str] = "images"
    video_key: Optional[str] = "videos"
    audio_key: Optional[str] = "audios"
    aux_alpha: Optional[float] = 0.1
    target_layer: Optional[int] = 28
    log_interest_freq: Optional[int] = 100
    
    sae_clustering_n_clusters: Optional[int] = 20
    sae_group_sparsity_coeff: Optional[float] = 0.0
    cluster_attn_strategy: Optional[str] = "all"
    cluster_spatial_coeff: Optional[float] = 0.02
    
    
@dataclass
class ModelArguments:
    model_path: str
    attn_implementation: str = "sdpa"
    model_cache_dir: Optional[str] = '~/.cache/huggingface'


@dataclass
class SaeConfig:
    sae_type: str = "TOPK_SAE"
    num_latents: int = 0
    expansion_factor: int = 32
    k: Optional[int] = 256
    dead_tokens_threshold: Optional[int] = 100000
    target_modules: Optional[str] = "model.layers.24.o_proj"
    warm_start: bool = False
    warm_start_path: Optional[str] = None
