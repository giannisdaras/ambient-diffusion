from transformers import T5EncoderModel
from pipeline_if import IFPipeline as DiffusionPipeline
from scheduling_ddpm import step
from diffusers.schedulers import DDPMScheduler
import gc
import torch
from dnnlib.util import save_images, save_image
import os
import math
from collections import OrderedDict
import numpy as np 
import dnnlib
import click
from tqdm import tqdm

def flush():
  gc.collect()
  torch.cuda.empty_cache()


@click.command()
@click.option('--upscale', is_flag=True, help='Upscale images', default=False)
@click.option('--checkpoint_path', type=str, help='Path to checkpoint file', required=True)
@click.option('--output_dir', type=str, help='Output directory', required=True)
@click.option('--batch_size', type=int, help='Batch size', default=16)
@click.option('--num_images', type=int, help='Number of images', default=50000)
@click.option('--batch_resume_index', type=int, help='Batch resume index', default=0)
@click.option('--save_collage', is_flag=True, help='Save collage', default=False)
@click.option('--save_separate', is_flag=True, help='Save separate', default=True)
@click.option('--corruption_probability', type=float, help='Corruption probability', default=0.8)
@click.option('--delta_probability', type=float, help='Delta probability', default=0.1)
@click.option('--corruption_pattern', type=str, help='Corruption pattern', default="dust")

def main(upscale, checkpoint_path, output_dir, batch_size, num_images, batch_resume_index, save_collage, save_separate, corruption_probability, delta_probability, corruption_pattern):
    os.makedirs(output_dir, exist_ok=True)
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        subfolder="text_encoder", 
        device_map="auto", 
        load_in_8bit=True, 
        variant="8bit"
    )

    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        text_encoder=text_encoder, # pass the previously instantiated 8bit text encoder
        unet=None, 
        device_map="auto"
    )


    prompts = batch_size * [""]
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompts)
    del text_encoder
    del pipe


    flush()
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        text_encoder=None, 
        variant="fp16", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    if checkpoint_path is not None:
        loaded_dict = torch.load(checkpoint_path, map_location="cuda")
        fixed_dict = OrderedDict({key.replace("_orig_mod.module.", ""): value for key, value in loaded_dict.items()})
        pipe.unet.load_state_dict(fixed_dict)
    
    pipe.scheduler.variance_type = None
    pipe.scheduler.config.variance_type = None
    pipe.scheduler.step = step.__get__(pipe.scheduler, DDPMScheduler)


    for batch_index in tqdm(range(int(np.ceil(num_images / batch_size)))):
        if batch_index < batch_resume_index:
            continue
        num_rows = int(math.sqrt(batch_size))
        pipe.safety_checker = None
        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds, 
            output_type="pt",
            corruption_probability=corruption_probability,
            delta_probability=delta_probability,
            guidance_scale=0.0,
            corruption_pattern=corruption_pattern,
            ).images

        if save_separate:
            for image_index, image in enumerate(images):
                save_image(image, os.path.join(output_dir, f"{batch_index}_{image_index}.png"))
        
        if save_collage:
            save_images(images, os.path.join(output_dir, f"{batch_index}.png"), num_rows=num_rows, num_cols=num_rows)


    if upscale:
        del pipe
        flush()
        assert save_collage == False, "Cannot upscale collages for now"

        super_res_pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", 
            text_encoder=None, # no use of text encoder => memory savings!
            variant="fp16", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        c = dnnlib.EasyDict()
        c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', 
                                        path=output_dir, use_labels=False, xflip=False, cache=False, 
                                        corruption_probability=0.0, delta_probability=0.0)
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs) # subclass of training.dataset.Dataset
        c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, num_replicas=1, rank=0, seed=42, shuffle=False)

        dataset_iterator = iter(
            torch.utils.data.DataLoader(
                dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_size, **c.data_loader_kwargs)
            )
        with torch.no_grad():
            for iter_index, dataset_iter in enumerate(dataset_iterator):
                image = dataset_iter[0]
                image = torch.tensor(image, device='cuda').to(torch.float32)
                super_res_images = super_res_pipe(
                    image=image, 
                    prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=negative_embeds,
                    guidance_scale=0.0,
                    output_type="pt",
                ).images
            if save_separate:
                for image_index, image in enumerate(super_res_images):
                    save_image(image, os.path.join(output_dir, f"{batch_index}_{image_index}_super_res.png"))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------