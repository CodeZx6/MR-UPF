import torch
import torch.nn as nn
from collections import OrderedDict


class LayerNormalization(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        original_dtype = x.dtype
        normalized = super().forward(x.type(torch.float32))
        return normalized.type(original_dtype)


class QuickGELUActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, attn_mask=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = LayerNormalization(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(dim, dim * 4)),
            ("activation", QuickGELUActivation()),
            ("linear2", nn.Linear(dim * 4, dim))
        ]))
        self.norm2 = LayerNormalization(dim)
        self.attn_mask = attn_mask
    
    def _compute_attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attention(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x):
        x = x + self._compute_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, width, depth, num_heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.depth = depth
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(width, num_heads, attn_mask) 
            for _ in range(depth)
        ])
    
    def forward(self, x):
        return self.blocks(x)


class TextualEncoder(nn.Module):
    def __init__(self, embed_dim, context_len, vocab_size, 
                 transformer_width, transformer_heads, transformer_layers):
        super().__init__()
        self.context_length = context_len
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        
        self.transformer = TransformerStack(
            width=transformer_width,
            depth=transformer_layers,
            num_heads=transformer_heads,
            attn_mask=self._build_causal_mask()
        )
        
        self.final_norm = LayerNormalization(transformer_width)
        self.projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.depth) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        
        for block in self.transformer.blocks:
            nn.init.normal_(block.attention.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attention.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.linear1.weight, std=fc_std)
            nn.init.normal_(block.mlp.linear2.weight, std=proj_std)
        
        if self.projection is not None:
            nn.init.normal_(self.projection, std=self.transformer.width ** -0.5)
    
    def _build_causal_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    def forward(self, text_tokens):
        token_embeds = self.token_embedding(text_tokens)
        token_embeds = token_embeds + self.positional_embedding
        
        token_embeds = token_embeds.permute(1, 0, 2)
        encoded = self.transformer(token_embeds)
        encoded = encoded.permute(1, 0, 2)
        encoded = self.final_norm(encoded)
        
        sequence_features = encoded[torch.arange(encoded.shape[0]), text_tokens.argmax(dim=-1)]
        projected = sequence_features @ self.projection
        
        return projected
