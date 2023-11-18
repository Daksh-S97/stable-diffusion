import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads, emb_dim, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        self.in_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=out_proj_bias)
        self.heads = n_heads
        self.d_head = emb_dim // n_heads

    def forward(self, x: torch.Tensor, mask = False):
        input_shape = x.shape
        b, s, d = input_shape

        interim_shape = (b, s, self.heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        attn_scores = q @ k.transpose(-1, -2)

        if mask:
            # upper diagonal matrix mask of -inf
            causal_mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
            attn_scores.masked_fill_(causal_mask, -torch.inf)

        attn_scores /= math.sqrt(self.d_head)

        attn_scores = F.softmax(attn_scores, dim=-1)
        out  = attn_scores @ v
        out = out.transpose(1,2)
        out = out.reshape(input_shape)

        return self.out_proj(out) 


class CrossAttention(nn.Module):

    def __init__(self, heads, d_embed, d_txt, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.w_k = nn.Linear(d_txt, d_embed, bias=in_proj_bias)
        self.w_v = nn.Linear(d_txt, d_embed, bias=in_proj_bias)
        self.w_o = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.heads = heads
        self.d_head = d_embed // heads

    def forward(self, x, txt):
        """
        x (img) - (b, s, d)
        txt (prompt) - (b, s2, d2) = (b, 77, 768)
        """
        inp_shape = x.shape
        b, s, d = inp_shape
        interim_shape = (b, -1, self.heads, self.d_head)

        q = self.w_q(x)
        k = self.w_k(txt)
        v = self.w_v(txt)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        attn_scores = q @ k.transpoe(-1,-2)
        attn_scores /= math.sqrt(self.d_head)

        attn_scores = F.softmax(attn_scores, dim=-1)
        out = attn_scores @ v
        out = out.transpose(1,2).contiguous()
        out = out.reshape(inp_shape)

        return self.w_o(out)
