import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.lin2 = nn.Linear(4 * emb_dim, 4 * emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.silu(x)
        return self.lin2(x)

class Upsample(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # h,w doubled
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x) 

class U_NetOut(nn.Module):

    def __init__(self, in_c, out_c) -> None:
        super().__init__() 
        self.norm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        return self.conv(x)   

class UNETResidualBlk(nn.Module):

    def __init__(self, in_c, out_c, time_emb_size=1280) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.lin_time = nn.Linear(time_emb_size, out_c)

        if in_c == out_c:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)    
    
    def forward(self, img, time):
        
        residue = img

        img = self.norm1(img)
        img = F.silu(img)
        img = self.conv1(img)
        time = F.silu(time)
        time = self.lin_time(time)

        agg = img + time.unsqueeze(-1).unsqueeze(-1)
        agg = self.norm2(agg)
        agg = F.silu(agg)
        agg = self.conv2(agg)

        return agg + self.residual_layer(residue)
    

class UNETAttnBlk(nn.Module):

    def __init__(self, heads, emb_dim, d_txt=768) -> None:
        super().__init__()
        channels = heads * emb_dim
        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_inp = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm1 = nn.LayerNorm(channels)
        self.attn_1 = SelfAttention(heads, channels, in_proj_bias=False)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attn_2 = CrossAttention(heads, channels, d_txt, in_proj_bias=False)
        self.layer_norm3 = nn.LayerNorm(channels)
        self.lin1 = nn.Linear(channels, 4 * channels * 2)
        self.lin2 = nn.Linear(4 * channels, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, img, txt):
        """
        img: (b, c, h, w)
        txt: (b, s, d)
        """ 
        residue_long = img
        img = self.groupnorm(img)
        img = self.conv_inp(img)

        n, c, h, w = img.shape
        img = img.view((n, c, h*w))
        img = img.transpose(1,2)

        residue_short = img
        img = self.layer_norm1(img)
        self.attn_1(img)
        img += residue_short

        residue_short = img
        img = self.layer_norm2(img)
        self.attn_2(img, txt)
        img += residue_short

        residue_short = img
        img = self.layer_norm3(img)
        img, gate = self.lin1(img).chunk(2, dim=-1)
        img = img * F.gelu(gate)
        img = self.lin2(img)

        img += residue_short

        img = img.transpose(1,2)
        img = img.view((n, c, h, w))

        return self.conv_out(img) + residue_long



class SwitchSequential(nn.Sequential):

    def forward(self, img, txt, time):
        for layer in self:
            if isinstance(layer, UNETAttnBlk):
                img = layer(img, txt)
            elif isinstance(layer, UNETResidualBlk):
                img = layer(img, time)
            else:
                img = layer(img)

        return img 


class U_Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNETResidualBlk(320, 320), UNETAttnBlk(8, 40)),
            SwitchSequential(UNETResidualBlk(320, 320), UNETAttnBlk(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlk(320, 640), UNETAttnBlk(8, 80)),
            SwitchSequential(UNETResidualBlk(640, 640), UNETAttnBlk(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlk(640, 1280), UNETAttnBlk(8, 160)),
            SwitchSequential(UNETResidualBlk(1280, 1280), UNETAttnBlk(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlk(1280, 1280)),

            SwitchSequential(UNETResidualBlk(1280, 1280))
            
        ])

        self.bottleneck = SwitchSequential(
            UNETResidualBlk(1280, 1280),
            UNETAttnBlk(8, 160),
            UNETResidualBlk(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNETResidualBlk(2560, 1280)),
            SwitchSequential(UNETResidualBlk(2560, 1280)),
            SwitchSequential(UNETResidualBlk(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNETResidualBlk(2560, 1280), UNETAttnBlk(8, 160)),
            SwitchSequential(UNETResidualBlk(2560, 1280), UNETAttnBlk(8, 160)),
            SwitchSequential(UNETResidualBlk(1920, 1280), UNETAttnBlk(8, 160), Upsample(1280)),
            
            SwitchSequential(UNETResidualBlk(1920, 640), UNETAttnBlk(8, 80)),
            SwitchSequential(UNETResidualBlk(1280, 640), UNETAttnBlk(8, 80)),
            SwitchSequential(UNETResidualBlk(960, 640), UNETAttnBlk(8, 80), Upsample(640)),

            SwitchSequential(UNETResidualBlk(960, 320), UNETAttnBlk(8, 40)),
            SwitchSequential(UNETResidualBlk(640, 320), UNETAttnBlk(8, 40)),
            SwitchSequential(UNETResidualBlk(640, 320), UNETAttnBlk(8, 40)),
        ])

class Diffusion(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(320)
        self.unet = U_Net()
        self.out_layer = U_NetOut(320,4)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, time: torch.Tensor):
        """
        img: img latents from encoder (b, 4, h/8, h/8)
        txt: text embeddings from clip (b, s, d)
        time: (1, 320)
        """ 
        # (1, 320) -> (1, 1280)
        time = self.time_embed(time)

        # (b, 4, h/8, h/8) -> (b, 320, h/8, h/8)
        out = self.unet(img, txt, time) 
        # (b, 320, h/8, w/8) -> (b, 4, h/8, w/8)
        out = self.out_layer(out)

        return out
