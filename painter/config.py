from dataclasses import dataclass

import torch


@dataclass
class PainterConfig:
    vocab_size: int = 50304  # TikToken GPT-2 vocab (50257) padded to nearest multiple of 64
    n_layers: int = 12
    n_embed: int = 768
    ffn_factor: int = 4
    context_len: int = 1024
    n_heads: int = 12
    d_head: int = 64  # n_embed // n_heads
    dropout: float = 0.2

    n_pred_heads: int = 4
    tie_weights: bool = True
    loss_discount_factor: float = 0.5

    def __post_init__(self):
        assert self.n_embed % self.n_heads == 0, "n_embed must be divisible by n_heads"
        self.d_head = self.n_embed // self.n_heads

@dataclass
class TrainerConfig:
    data_dir: str = "data/fineweb_edu"

    batch_size: int = 64
    gradient_accumulation_steps: int = 4

    learning_rate: float = 3e-4
    min_lr: float = 6e-5
    epochs: int = 60 

    weight_decay: float = 1e-4 
    grad_clip: float = 1.0

    log_dir: str = "logs"
    eval_steps: int = 100
    save_interval: int = 5000

    scheduler: str = "cosine"
    warmup_epochs: int = 10
    warmup_factor: float = 0.01

    checkpoint_dir: str = "checkpoints"

    compile_model: bool = True

    num_dataloader_workers: int = 4

