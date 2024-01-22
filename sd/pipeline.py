import torch
import numpy as np 
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(prompt:str,
             uncond_prompt:str,#neg prompt
             input_image=None,
             strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name='ddpm', n_inference_steps=50, 
             models={}, 
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None,
             anim=False
             ):

    with torch.no_grad():
        if not (0<strength<=1):
                raise ValueError("strength must be between 0 and 1")
        
        # if idle_device:
        #     #print("here")
        #     to_idle: lambda x: x.to(idle_device)
        # else:
        #     to_idle: lambda x:x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids #convert prompt to tokens
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device) #(bsize,seqlen)
            cond_context = clip(cond_tokens) #(bsize,seqlen,dim)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids #convert prompt to tokens
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device) #(bsize,seqlen)
            uncond_context = clip(uncond_tokens) #(bsize,seqlen,dim)

            context = torch.cat([cond_context,uncond_context]) #(2,seqlen,dim) = (2,77,768)
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids #convert prompt to tokens
            tokens = torch.tensor(tokens, dtype=torch.long, device=device) #(bsize,seqlen)
            context = clip(tokens) #(bsize,seqlen,dim) = (1,77,768)
        
        #to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"unkown sampler {sampler_name}")

        latents_shape = (1,4,LATENT_HEIGHT,LATENT_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH,HEIGHT))
            input_image_tensor = np.array(input_image_tensor) #(h,w,ch)
            input_image_tensor = torch.tensor(input_image_tensor,dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
            input_image_tensor = input_image_tensor.unsqueeze(0) #(bsize,h,w,ch)
            input_image_tensor = input_image_tensor.permute(0,3,1,2) #(bsize,ch,h,w)

            encoder_noise = torch.randn(latents_shape,generator=generator,device=device)

            latents = encoder(input_image_tensor,encoder_noise) #run image thru VAE encoder
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            #to_idle(encoder)
        else:
            latents = torch.randn(latents_shape,generator=generator,device=device) #for text to img start w/ random noise
        

        diffusion = models["diffusion"]
        diffusion.to(device)

        latent_arr = []
        timesteps = tqdm(sampler.timesteps)
        for i,timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device) #(1,320)
            model_input = latents #(bsize,4,latents_h,latents_w), (latents_h same as l_h)

            if do_cfg:
                model_input = model_input.repeat(2,1,1,1) #(bsize,4,l_h,l_w) -> (2*bsize,4,l_h,l_w)
            
            model_output = diffusion(model_input,context,time_embedding) #noise pred by UNET

            if do_cfg:
                output_cond,output_uncond = model_output.chunk(2)
                model_output = cfg_scale*(output_cond-output_uncond) + output_uncond
            
            latents = sampler.step(timestep,latents,model_output) #remove noise pred by UNET
            latent_arr.append(latents)
            

        #to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        img_arr = []
        #to_idle(decoder)
        if(anim):
            for i in range(len(latent_arr)):
                img = decoder(latent_arr[i])
                img = rescale(img,(-1,1),(0,255),clamp=True)
                img = img.permute(0,2,3,1)
                img = img.to("cpu",torch.uint8).numpy()
                img_arr.append(img)
        else:
            images = decoder(latents)
            images = rescale(images,(-1,1),(0,255),clamp=True)
            images = images.permute(0,2,3,1) #(bs,ch,h,w) -> (bs,h,w,ch)
            images = images.to("cpu",torch.uint8).numpy()
            img_arr.append(images)

        #return images[0]
        print(seed)
        return img_arr
    
def rescale(x,old_range,new_range,clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max-new_min)/(old_max-old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min,new_max)
    return x

def get_time_embedding(timestep):
    
    freqs = torch.pow(10000, -torch.arange(start=0,end=160, dtype=torch.float32)/160) #(160,)
    x = torch.tensor([timestep],dtype=torch.float32)[:,None] * freqs[None] #(1,160)
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1) #(1,320)














            






