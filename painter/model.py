import torch
from torch import nn
import torch.nn.functional as F
from config import PainterConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: PainterConfig):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            config.n_embed,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * config.ffn_factor),
            nn.GELU(),
            nn.Linear(config.n_embed * config.ffn_factor, config.n_embed),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
    
    def forward(self, x):
        norm = self.norm1(x)
        attn, _ = self.attn(norm, norm, norm, is_causal=True, need_weights=False)
        x = x + self.dropout(attn)
        x = x + self.ffn(self.norm2(x))
        return x

        
class MultiTokenPredictionBlock(nn.Module):
    def __init__(self, config: PainterConfig):
        super().__init__()
        self.proj = nn.Linear(2 * config.n_embed, config.n_embed)
        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.transformer = TransformerBlock(config)
    
    def forward(self, h_prev, h_next):
        """
        Args:
            h_prev: (B, T, n_embed) - hidden state from previous depth
            h_next: (B, T, n_embed) - embedding of future token
        """
        h_prev = self.norm1(h_prev)
        h_next = self.norm2(h_next)
        
        h = torch.cat([h_prev, h_next], dim=-1)  # (B, T, 2*n_embed)
        h = self.proj(h)  # (B, T, n_embed)
        
        h = self.transformer(h)  # (B, T, n_embed)
        
        return h


class Painter(nn.Module):
    def __init__(self, config: PainterConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.context_len, config.n_embed)
        
        self.doc_layer = nn.Sequential(
            nn.Linear(1, config.n_embed),
            nn.GELU(),
            nn.Linear(config.n_embed, config.n_embed),
            nn.LayerNorm(config.n_embed)
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.n_embed)
        
        self.mtps = nn.ModuleList([
            MultiTokenPredictionBlock(config) for _ in range(config.n_pred_heads - 1)
        ])
        
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        self.register_buffer(
            'position_ids',
            torch.arange(0, config.context_len).unsqueeze(0)
        )
        self.dropout = nn.Dropout(config.dropout)
        
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if config.tie_weights:
            self.head.weight = self.token_embedding.weight
    
    def forward(self, x, targets=None, n_pred_heads=None):
        """
        Args:
            x: (B, T, 2) - tokens and document positions
            n_pred_heads: Number of prediction depths (None = use all)
        Returns:
            During training: (B, T, n_depths, vocab_size)
            During inference: (B, T, vocab_size)
        """
        B, T, K = x.shape
        assert K == 2, "Input dimension needs to be B, T, 2"
        
        if n_pred_heads is None:
            n_pred_heads = self.config.n_pred_heads
        
        assert n_pred_heads > 0, "n_pred_heads must be > 0"
        assert n_pred_heads <= len(self.mtps) + 1, \
            f"n_pred_heads ({n_pred_heads}) must be <= {self.config.n_pred_heads}"
        
        position_embedding = self.position_embedding(self.position_ids[:, :T])  # (B, T, n_embed)
        token_embedding = self.token_embedding(x[:, :T, 0].long())  # (B, T, n_embed)
        doc_pos = torch.log1p(x[:, :, 1])  # (B, T)
        doc_embedding = self.doc_layer(doc_pos.unsqueeze(-1))  # (B, T, n_embed)
        
        h = self.dropout(position_embedding + token_embedding + doc_embedding)
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm(h)  # (B, T, n_embed)
        
        logits_main = self.head(h)  # (B, T, vocab_size)
        
        if not self.training:
            return logits_main, None

        assert targets is not None, "For training, targets need to be specified."
        
        # Multi Token Prediction à la DeepSeek to learn deeper patterns
        predictions = [logits_main]
        
        mtps = self.mtps[:n_pred_heads - 1]
        
        h_prev = h
        
        for k, mtp in enumerate(mtps):
            if k + 1 >= T:
                break
            
            valid_pos = T - k

            # embed truncated sequence of next tokens
            x_next = targets[:, k:].long()  # (B, T-k)
            h_next = self.token_embedding(x_next)  # (B, T-k, n_embed)
            
            h_prev = h_prev[:, :valid_pos, :]  # (B, T-k, n_embed)
            
            h_prev = mtp(h_prev, h_next)  # (B, T-k, n_embed)
            
            logits_k = self.head(h_prev)  # (B, T-k, vocab_size)
            
            logits_k = F.pad(
                logits_k,
                (0, 0, 0, k),
                value=0
            )  # (B, T, vocab_size)
            
            predictions.append(logits_k)
        
        logits = torch.stack(predictions, dim=2)  # (B, T, n_pred_heads, vocab_size)
        loss = self._compute_loss(logits, targets)
        
        return logits, loss
    
    def _compute_loss(self, logits, targets):
        """Multi-token prediction loss with discounting"""
        B, T, n_heads, V = logits.shape  # batch_size, context_len, n_heads, vocab_size

        weights = torch.tensor(
            [self.config.loss_discount_factor ** i for i in range(n_heads)],
            device=targets.device
        )
        
        total_loss = 0.0
        total_count = 0
        
        for k in range(n_heads):
            head_logits = logits[:, :, k, :]  # B, T, V
            
            # targets are shifted by 1, so the target at T - 1 is the token at index T 
            # therefore the last valid prediction is by T - k 
            # which predicts token index T which in turn is targets[:, T - 1]
            valid_pos = T - k
            
            if valid_pos <= 0:
                continue
            
            valid_logits = head_logits[:, :valid_pos, :]  # B, valid_pos, V
            valid_targets = targets[:, k:] # B, valid_pos
            
            loss = F.cross_entropy(
                valid_logits.reshape(-1, V),
                valid_targets.reshape(-1),
                reduction='sum' 
            )
            
            total_loss += weights[k] * loss
            total_count += valid_logits.numel() / V
        
        weighted_loss = total_loss / total_count if total_count > 0 else total_loss
        
        return weighted_loss
