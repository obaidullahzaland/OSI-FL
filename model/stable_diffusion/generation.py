# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
from .ptp_utils import get_word_inds, text_under_image, view_images, register_attention_control, init_latent, latent2image, diffuse_reconstruction_step, calculate_distortion
from torch.optim.adam import Adam
from PIL import Image
import matplotlib.pyplot as plt

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5 # 7.5 when editing
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    sub_category_embedding: Optional[torch.FloatTensor] = None,
    sub_category: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    save_path = None,
):
    batch_size = len(prompt)
    height = width = 512    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    text_embeddings_all = sub_category_embedding.to(model.device)    #sjin: should be replaced with standard + difference, since I don't have the difference now I will use the standard
    # text_embeddings = text_embeddings_all[1].unsqueeze(0)
    text_embeddings = torch.mean(text_embeddings_all, dim=0).unsqueeze(0)
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            # print("unconditional embeddings.shape", uncond_embeddings_.shape)
            # print("text_embeddings.shape", text_embeddings.shape)
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffuse_reconstruction_step(model, latents, context=context, t=t, guidance_scale=GUIDANCE_SCALE, low_resource=False)
            
        
    if return_type == 'image':
        image = latent2image(model.vae, latents, save_path)
    else:
        image = latents
    return image, latent, None



def run_and_display(prompts=None, sub_category_embedding=None, sub_category=None, generator=None, latent=None, uncond_embeddings=None,image_encoded=None, save_path=None):
    images, x_t, distortion_at_all_timesteps = text2image_ldm_stable(ldm_stable, prompts, sub_category_embedding=sub_category_embedding, sub_category=sub_category,latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, start_time = NUM_DDIM_STEPS, save_path=save_path)
    return images, x_t, distortion_at_all_timesteps


print(f"Using {device}.")

def generate(standard_category=None, sub_category=None, sub_category_embedding=None, seed=None, save_path = None):
    x_t = None
    uncond_embeddings = None
    image_enc = None  
    prompts = [standard_category]
    g_cpu = torch.Generator().manual_seed(seed)
    image_inv, x_t, distortion_at_all_timesteps = run_and_display(prompts=prompts, sub_category_embedding=sub_category_embedding, sub_category=sub_category, generator=g_cpu, latent=x_t, uncond_embeddings=uncond_embeddings,image_encoded=image_enc, save_path=save_path)
    #save the image
    # plt.savefig(image_inv, save_path)
    # return cross_attn_words_timesteps, distortion_at_all_timesteps