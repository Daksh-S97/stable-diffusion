import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbeds(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, n_tokens: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(n_tokens, emb_dim))

    def forward(self, x):
        x = self.embedding(x)
        x += self.pos_embedding
        return x 

class CLIP_Layer(nn.Module):

    def __init__(self, heads: int, emb_dim: int) -> None:
        super().__init__()       
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attention = SelfAttention(heads, emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.lin1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.lin2 = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.norm1(x)
        x = self.attention(x, causal_mask = True)
        x += residue

        residue = x
        x = self.norm2(x)
        x = self.lin1(x)
        x = x * torch.sigmoid(1.702 * x) # quickGeLU activation
        x = self.lin2(x)
        x += residue

        return x    

class CLIP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbeds(49408, 768, 77)   # vocab size, emb_dim, seq_len
        self.layers = nn.ModuleList([CLIP_Layer(12, 768) for i in range(12)])
        self.norm  = nn.LayerNorm(768)


    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # x: (b, s)
        x = x.type(torch.long)

        x = self.embedding(x) # (b,s,d)

        for layer in self.layers:
            x = layer(x)

        out = self.norm(x)

        # (b,s,d)
        return out    

