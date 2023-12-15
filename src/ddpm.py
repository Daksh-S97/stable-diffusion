import torch 
import numpy as np


class DDPMSampler:

    def __init__(self, generator: torch.Generator, train_steps=1000, beta_start=0.00085, beta_end=0.012):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, train_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.train_steps = train_steps
        self.timesteps = torch.from_numpy(np.arange(0,self.train_steps)[::-1].copy())

    
    def set_inference_steps(self, inf_steps=50):
        self.inf_steps = inf_steps
        ratio = self.train_steps // self.inf_steps
        inf_timesteps = self.timesteps[::ratio]
        self.timesteps = inf_timesteps

    def set_strength(self, strength=1):
        start = self.inf_steps - int(self.inf_steps * strength)
        self.timesteps = self.timesteps[start:]
        self.start = start            


    def step(self, timestep: int, latents: torch.Tensor, pred_noise: torch.Tensor):
        prev_t = timestep - (self.train_steps // self.inf_steps)
        cum_alpha_t = self.alpha_cum[timestep]
        cum_alpha_prev = self.alpha_cum[prev_t] if prev_t >= 0 else self.one
        cum_beta_t = 1 - cum_alpha_t
        cum_beta_prev = 1 - cum_alpha_prev
        curr_alpha = cum_alpha_t / cum_alpha_prev
        curr_beta = 1 - curr_alpha

        # image at t=0 i.e the original image
        og_img = (latents - cum_beta_t ** 0.5 * pred_noise) / cum_alpha_t ** 0.5
        og_img_coeff = (cum_alpha_prev ** 0.5 * curr_beta) / cum_beta_t
        curr_coeff = (curr_alpha ** 0.5 * cum_beta_prev) / cum_beta_t

        # prev sample mean
        prev_sample_mean = og_img_coeff * og_img + curr_coeff * latents

        var = curr_beta * (1 - cum_alpha_prev) / (1 - cum_alpha_t)
        var = torch.clamp(var, min=1e-20)
        variance = 0
        if timestep > 0:
            device = pred_noise.device
            noise = torch.randn(pred_noise.shape, generator=self.generator, device=device, dtype=pred_noise.dtype)
            variance = (var ** 0.5) * noise  #stddev * noise

        # sample from new distribution
        prev_sample = prev_sample_mean + variance

        return prev_sample
    

    def add_noise(self, original_img: torch.FloatTensor, timestep: torch.IntTensor):
        alpha_cum = self.alpha_cum.to(device=original_img.device, dtype=original_img.dtype)
        time = timestep.to(original_img.device)

        # alphaT is actually alpha_bar_T (cumulative alpha_t)
        sqrt_alphaT = alpha_cum[time] ** 0.5            # for mean
        sqrt_alphaT = sqrt_alphaT.flatten()
        while len(sqrt_alphaT.shape) < len(original_img.shape):
            sqrt_alphaT = sqrt_alphaT.unsqueeze(-1)

        sqrt_one_minus_alphaT = (1.0 - alpha_cum[time]) ** 0.5       # stddev
        sqrt_one_minus_alphaT = sqrt_one_minus_alphaT.flatten()
        while len(sqrt_one_minus_alphaT.shape) < len(original_img.shape):
            sqrt_one_minus_alphaT = sqrt_one_minus_alphaT.unsqueeze(-1)

        noise = torch.randn(original_img.shape, generator=self.generator, device=original_img.device, dtype=original_img.shape)
        noisy_samples = (sqrt_alphaT * original_img) * (sqrt_one_minus_alphaT) * noise 
        return noisy_samples       
