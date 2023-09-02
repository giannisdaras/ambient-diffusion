import dnnlib
from dnnlib.util import load_image, pad_image
import click
import torch
from torch_utils import misc
from torch_utils import distributed as dist
from tqdm import tqdm
from dnnlib.util import pad_image, is_file, save_image
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt


@click.command()
@click.option('--input_dir', 'input_dir',  help='Location of the folder where the network outputs are stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--output_dir', 'output_dir',  help='Location of the folder where the outputs should be stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--features_path', help='Path to save/load dataset features from', metavar='PATH|URL', type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--max_size', help='Limit training samples.', type=int, default=None, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int, default=42)
@click.option('--normalize', help='Whether to normalize feature vectors', metavar='BOOL', type=bool, default=True, show_default=True)


def main(input_dir, output_dir, features_path, data, max_size, cache, workers, batch, batch_gpu, seed, normalize):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if seed is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        seed = int(seed)
    # Select batch size per GPU.
    batch_gpu_total = batch // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    

    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=False, xflip=False, cache=cache, 
                                        corruption_probability=0.0, delta_probability=0.0)
    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs) # subclass of training.dataset.Dataset
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to('cuda')
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[torch.device('cuda')], broadcast_buffers=False)


    if not is_file(features_path):

        c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=workers, prefetch_factor=2)
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank(), seed=seed, shuffle=False)

        dataset_iterator = iter(
            torch.utils.data.DataLoader(
                dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **c.data_loader_kwargs)
            )

        features = []
        with torch.no_grad():
            for dataset_iter in tqdm(dataset_iterator):
                images = dataset_iter[0]
                images = images.to('cuda').to(torch.float32)
                local_features = feature_extractor((pad_image(images) + 1) / 2. ).cpu()
                features.append(local_features)
        features = np.concatenate(features)
        np.save(features_path, features)
    else:
        features = np.load(features_path)
    
    # normalize dataset features
    if normalize:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

    os.makedirs(output_dir, exist_ok=True)
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=input_dir, use_labels=False, xflip=False, cache=cache, 
                                        corruption_probability=0.0, delta_probability=0.0, max_size=max_size)
    outputs_dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs) # subclass of training.dataset.Dataset
    
    max_products = []
    softmax_products = []
    with torch.no_grad():
        for iter_index, dataset_iter in enumerate(tqdm(outputs_dataset_obj)):
            images = dataset_iter[0]
            images = torch.tensor(images, device='cuda').to(torch.float32).unsqueeze(0)
            save_image(images[0], os.path.join(output_dir, f'{iter_index}_dataset_image_{iter_index}.png'))

            local_features = feature_extractor((pad_image(images) + 1) / 2. ).cpu()
            if normalize:
                local_features /= np.linalg.norm(local_features, axis=1, keepdims=True)

            products = (local_features @ features.T).squeeze()

            # get normalized probabilities from logits
            softmax_products.append(torch.nn.functional.softmax(products).max())
            max_products.append(float(products.max()))
            sorted_indices = products.argsort().tolist()[::-1]
            for k in range(3):
                i = sorted_indices[k]
                images = torch.tensor(dataset_obj[i][0])
                
                save_image(images, os.path.join(output_dir, f'{iter_index}_nearest_neighbors_{k}_dataset_index_{i}.png'))
            
            
    with open(os.path.join(output_dir, 'max_products.pkl'), 'wb') as f:
        pickle.dump(max_products, f)
    
    with open(os.path.join(output_dir, 'softmax_products.pkl'), 'wb') as f:
        pickle.dump(softmax_products, f)


if __name__ == '__main__':
    main()
        
