import gc
import torch
import os
from collections import OrderedDict
import numpy as np 
import click
from deepfloyd_if.pipelines import super_resolution
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.modules import IFStageII
from PIL import Image


def flush():
  gc.collect()
  torch.cuda.empty_cache()


@click.command()
@click.option('--input_dir', type=str, help='Input directory', required=True)
@click.option('--output_dir', type=str, help='Output directory', required=True)
@click.option('--batch_size', type=int, help='Batch size', default=16)
@click.option('--num_images', type=int, help='Number of images', default=50000)
@click.option('--batch_resume_index', type=int, help='Batch resume index', default=0)
@click.option('--save_collage', is_flag=True, help='Save collage', default=False)
@click.option('--save_separate', is_flag=True, help='Save separate', default=True)
@click.option('--corruption_probability', type=float, help='Corruption probability', default=0.8)
@click.option('--delta_probability', type=float, help='Delta probability', default=0.1)
@click.option('--corruption_pattern', type=str, help='Corruption pattern', default="dust")
@click.option('--device', type=str, help='cpu|cuda', default="cuda")

def main(input_dir, output_dir, batch_size, num_images, batch_resume_index, save_collage, save_separate, corruption_probability, delta_probability, 
         corruption_pattern, device):
    os.makedirs(output_dir, exist_ok=True)

    t5 = T5Embedder(device=device)
    if_II = IFStageII('IF-II-L-v1.0', device=device)
    if_II.model.load_state_dict(torch.load("/raid/id4439/up_finetuning_06/state_dict_2250.pt"))


    # iterate over all images in input dir
    for image_name in os.listdir(input_dir):
        print(image_name)
        raw_pil_image = Image.open(os.path.join(input_dir, image_name))
        middle_res = super_resolution(
            t5,
            if_III=if_II,
            prompt=[''],
            support_pil_img=raw_pil_image,
            img_scale=4.,
            img_size=64,
            disable_watermark=True,
            if_III_kwargs={
                'sample_timestep_respacing': 'smart100',
                'aug_level': 0.0,
                'guidance_scale': 6.0,
            },
        )["III"][0]
        middle_res.save(os.path.join(output_dir, image_name))
        print("saved")
    




#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------