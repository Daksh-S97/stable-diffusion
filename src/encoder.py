import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAEAttnBlk, VAEResidualBlk

# conv formula
# dim_out = floor((dim_in - k_size + 2 * pad)/stride) + 1

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super.__init__(
            # (B, 3, H, W) ->  (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # residual blk doesn't change h or w
            # arguments -> in_channels, out_channels
            VAEResidualBlk(128, 128),
            VAEResidualBlk(128, 128),
            
            # (b, 128, h, w)- > (b, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAEResidualBlk(128, 256),
            VAEResidualBlk(256, 256),

            # (h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAEResidualBlk(256, 512),
            VAEResidualBlk(512, 512),

            # (h/8, h/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAEResidualBlk(512, 512),
            VAEResidualBlk(512, 512),

            VAEResidualBlk(512, 512),
            VAEAttnBlk(512),
            VAEResidualBlk(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # (b, 512, h/8, w/8) -> (b, 8, h/8, w/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x -> (b, c, h, w)
        # noise -> (b, out_c, h/8, w/8)

        for module in self:
            if getattr(module, 'stride') == (2,2):
                # for these conv layers, we only want padding on right and bottom
                x = F.pad(x, (0,1,0,1))
            x = module(x) 

        # (b, 8, h/8, w/8) -> 2 tensors of shape (b, 4, h/8, h/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # restrict values to the provide range
        log_variance = torch.clamp(log_variance, -30, 20)       
        variance = log_variance.exp()
        stddev = variance.sqrt()

        # Z (N(0,1)) -> X((mu, sigma)) => x = mu + sigma * z
        # sample from distribution with given mu and sigma by using the normal distribution
        x = mean + stddev * noise

        #arbitrary scaling lol
        x *= 0.18215

        # (b, 4, h/8, h/8)
        return x