import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = 512 // 8

def generate(prompt: str, neg_prompt: str, input_img=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler="ddpm", inf_steps=50, models={},
             seed=None, device=None, idle_device=None, tokenizer=None):
    """
    neg_prompt: specify objects you don't want in the image; empty string for cfg
    cfg_scale: weight for cfg signal
    idle_device: offload the models which are not in use to this device
    strength: how much attention is given to input image while generating output image; more strength matlab less dependent on input image
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            return ValueError("strenght must bew between 0 and 1")

    if idle_device:
        to_idle: lambda x: x.to(idle_device)
    else:
        to_idle: lambda x: x 

    generator = torch.Generator(device=device) # RNG
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    clip = models["clip"]
    clip.to(device)                   

    if do_cfg:
        # tokenize
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device) # (b,s) = (b, 77)
        cond_ctxt = clip(cond_tokens) # (b, s, d) = (1, 77, 768)

        uncond_tokens = tokenizer.batch_encode_plus([uncond_tokens], padding = "max_length", max_length = 77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_ctxt = clip(uncond_tokens)

        ctxt = torch.cat([cond_ctxt, uncond_ctxt])

    else:
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device) # (b,s) = (b, 77)
        ctxt = clip(cond_tokens) # (b, s, d) = (1, 77, 768)
    
    to_idle(clip) # offload clip from current device since we're done with it  

    if sampler == "ddpm":
        samp = DDPMSampler(generator)  
        samp.set_inference_steps(inf_steps)
    else:
        raise ValueError("Unknown sampler")

    lat_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    if input_img:
       
       #image-to-image
       enc = models["encoder"]
       enc.to(device)

       input_img = input_img.resize([WIDTH, HEIGHT])
       input_img = np.array(input_img)
       inp_tensor = torch.tensor(input_img, dtype=torch.float32) # (h, w, c)
       inp_tensor = rescale(inp_tensor, (0, 255), (-1, 1)) 
       inp_tensor = inp_tensor.unsqueeze(0) #(b, h, w, c)
       inp_tensor = inp_tensor.permute(0, 3, 1, 2) # (b, c, h, w)

       enc_noise = torch.randn(lat_shape, generator=generator, device=device)
       latents = enc(inp_tensor, enc_noise)

       samp.set_strength(strength=strength)
       latents = samp.add_noise(latents, sampler.timesteps[0])
       
       to_idle(enc)

    else:
        #txt-to-image
        # random noise (N(0,1))
        latents = torch.randn(lat_shape, generator=generator, device=device)

    diffusion = models["diffusion"]
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)
    
    for i, time in enumerate(timesteps):
        time_embed = get_time_embeds(time).to(device) # (1,320)

        inp = latents
        if do_cfg:
            # duplicate the tensor to account for the unconditioned prompt
            inp = inp.repeat(2,1,1,1) #(2 * b, 4, h, w)

        # noise predicted by U-Net
        out = diffusion(inp, ctxt, time_embed) 
        if do_cfg:
            out_cond, out_uncond = out.chunk(2)
            out = cfg_scale * (out_cond - out_uncond) + out_uncond

        # remove noise      
        latents = sampler.step(time, latents, out)

    to_idle(diffusion)
    decoder = models["decoder"]
    img = decoder(latents)
    to_idle(decoder)

    img = rescale(img, (-1, 1), (0,255), clamp=True)
    img = img.permute(0,2,3,1)
    img = img.to("cpu", torch.uint8).numpy()
    return img[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x = (new_max - new_min) / (old_max - old_min)
    x += old_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x 

def get_time_embeds(timestep):
    # same as transformer pos_enc
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) # (1, 160)
    tens = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(tens), torch.sin(tens)]) # (1, 320)






