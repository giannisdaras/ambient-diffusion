import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_random_mask(image_shape, survival_probability, mask_full_rgb=False, same_for_all_batch=False, device='cuda', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if same_for_all_batch:
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape[1:]).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32).repeat([image_shape[0], 1, 1, 1])
    else:
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32)
    
    if mask_full_rgb:
        corruption_mask = corruption_mask[:, 0]
        corruption_mask = corruption_mask.repeat([3, 1, 1, 1]).transpose(1, 0)
    return corruption_mask


def get_box_mask(image_shape, survival_probability, same_for_all_batch=False, device='cuda'):
    """Creates a mask with a box of size survival_probability * image_shape[1] somewhere randomly in the image.
        Args:
            image_shape: (batch_size, num_channels, height, width)
            survival_probability: probability of a pixel being unmasked
            same_for_all_batch: if True, the same mask is applied to all images in the batch
            device: device to use for the mask
        Returns:
            mask: (batch_size, num_channels, height, width)
    """
    batch_size = image_shape[0]
    num_channels = image_shape[1]
    height = image_shape[2]
    width = image_shape[3]

    # create a mask with the same size as the image
    mask = torch.zeros((batch_size, num_channels, height, width), device=device)

    # decide where to place the box randomly -- set the box at a different location for each image in the batch
    box_start_row = torch.randint(0, height, (batch_size, 1, 1), device=device)
    box_start_col = torch.randint(0, width, (batch_size, 1, 1), device=device)
    box_height = torch.ceil(torch.tensor((1 - survival_probability) * height)).int()
    box_width = torch.ceil(torch.tensor((1 - survival_probability) * width)).int()
    
    
    # mask[:, :, box_start_row:box_start_row + box_height, box_start_col:box_start_col + box_width] = 1.0

    box_start_row_expanded = box_start_row.view(batch_size, 1, 1, 1)
    box_start_col_expanded = box_start_col.view(batch_size, 1, 1, 1)

    rows = torch.arange(height, device=device).view(1, 1, -1, 1).expand_as(mask)
    cols = torch.arange(width, device=device).view(1, 1, 1, -1).expand_as(mask)

    inside_box_rows = (rows >= box_start_row_expanded) & (rows < (box_start_row_expanded + box_height))
    inside_box_cols = (cols >= box_start_col_expanded) & (cols < (box_start_col_expanded + box_width))

    inside_box = inside_box_rows & inside_box_cols
    mask[inside_box] = 1.0
    
    return 1 - mask


def get_patch_mask(image_shape, crop_size, same_for_all_batch=False, device='cuda'):
    """
        Args:
            image_shape: (batch_size, num_channels, height, width)
            crop_size: probability of a pixel being unmasked
            same_for_all_batch: if True, the same mask is applied to all images in the batch
            device: device to use for the mask
        Returns:
            mask: (batch_size, num_channels, height, width)
    """
    batch_size = image_shape[0]
    num_channels = image_shape[1]
    height = image_shape[2]
    width = image_shape[3]

    # create a mask with the same size as the image
    mask = torch.zeros((batch_size, num_channels, height, width), device=device)

    max_x = width - crop_size
    max_y = height - crop_size
    box_start_row = torch.randint(0, max_x, (batch_size, 1, 1, 1), device=device)
    box_start_col = torch.randint(0, max_y, (batch_size, 1, 1, 1), device=device)

    rows = torch.arange(height, device=device).view(1, 1, -1, 1).expand_as(mask)
    cols = torch.arange(width, device=device).view(1, 1, 1, -1).expand_as(mask)

    inside_box_rows = (rows >= box_start_row) & (rows < (box_start_row + crop_size))
    inside_box_cols = (cols >= box_start_col) & (cols < (box_start_col + crop_size))
    inside_box = inside_box_rows & inside_box_cols
    mask[inside_box] = 1.0
    
    return mask


def get_hat_patch_mask(patch_mask, crop_size, hat_crop_size, same_for_all_batch=False, device='cuda'):
    hat_mask = get_patch_mask((patch_mask.shape[0], patch_mask.shape[1], crop_size, crop_size), hat_crop_size, same_for_all_batch=same_for_all_batch, device=device)
    patch_indices = torch.nonzero(patch_mask.view(-1) == 1).squeeze()
    expanded_hat_mask = hat_mask.view(-1).expand_as(patch_indices)
    hat_patch_mask = torch.clone(patch_mask)
    hat_patch_mask.view(-1)[patch_indices] = expanded_hat_mask
    hat_patch_mask = hat_patch_mask.reshape(patch_mask.shape)
    return hat_patch_mask



class ForwardOperator(nn.Module):
    """
        Base class for forward operators.
    """
    def corrupt(self, x, *args, **kwargs):
        raise NotImplementedError("Corrupt method not implemented for class {}".format(self.__class__.__name__))

    def hat_corrupt(self, x, *args, **kwargs):
        raise NotImplementedError("Hat corrupt method not implemented for class {}".format(self.__class__.__name__))
    

class MaskingForwardOperator(ForwardOperator):
    def __init__(self, corruption_probability, delta_probability, mask_full_rgb=True):
        self.corruption_probability = corruption_probability
        self.delta_probability = delta_probability
        self.mask_full_rgb = mask_full_rgb
    
    def corrupt(self, x, mask=None):
        """
            Args:
                x: (batch_size, num_channels, height, width): *clean* input images
                mask: (batch_size, num_channels, height, width): mask used to corrupt the input images. If None, it is generated randomly
            Returns:
                x_corrupted: (batch_size, num_channels, height, width): *corrupted* input images
                mask: (batch_size, num_channels, height, width): mask used to corrupt the input images
        """
        if mask is None:
            mask = get_random_mask(x.shape, 1 - self.corruption_probability, mask_full_rgb=self.mask_full_rgb, 
                                same_for_all_batch=False, device=x.device, seed=None)
        return x * mask, mask
    
    def hat_corrupt(self, x, mask=None, hat_mask=None):
        """
            Args:
                x: (batch_size, num_channels, height, width): *corrupted* input images
                mask: (batch_size, num_channels, height, width): mask used to corrupt the input images
                hat_mask: (batch_size, num_channels, height, width): mask for hat corruption. If None, it is generated randomly.
            Returns:
                x_hat: (batch_size, num_channels, height, width): *hat-corrupted* input images
                hat_mask: (batch_size, num_channels, height, width): mask used to hat-corrupt the input images
        """
        if mask is None:
            _, mask = self.corrupt(x)

        if hat_mask is None:
            hat_mask = get_random_mask(x.shape, 1 - self.delta_probability, mask_full_rgb=self.mask_full_rgb, 
                                        same_for_all_batch=False, device=x.device, seed=None)
        hat_mask = mask * hat_mask
        return x * hat_mask, hat_mask


class AveragingForwardOperator(ForwardOperator):
    def __init__(self, corruption_probability, downsampling_factor=8):
        self.corruption_probability = corruption_probability
        self.downsampling_factor = downsampling_factor
    
    def corrupt(self, x, mask=None):
        """
            Args:
                
                
        """        
        if mask is None:
            # create a bernoulli mask to decide which images in the batch will get downsampled
            mask = torch.bernoulli(torch.ones(x.shape[0], device=x.device) * self.corruption_probability)
            mask = mask.view(-1, 1, 1, 1)
        
        # downsample all images
        downsampled_images = F.avg_pool2d(x, self.downsampling_factor)
        corrupted_images = F.interpolate(downsampled_images, size=(x.shape[2], x.shape[3]), mode='nearest')        
        corrupted_images = mask * corrupted_images + (1 - mask) * x
        return corrupted_images, mask
    
    def hat_corrupt(self, x, mask=None, *args):
        if mask is None:
            _, mask = self.corrupt(x)
        # downsample all images that are not already downsampled
        return self.corrupt(x, torch.ones_like(mask))[0], torch.ones_like(mask)


def get_operator(corruption_pattern, corruption_probability=None, delta_probability=None, downsampling_factor=None):
    if corruption_pattern == "dust":
        assert corruption_probability is not None, "corruption_probability must be specified for dust corruption pattern"
        assert delta_probability is not None, "delta_probability must be specified for dust corruption pattern"
        return MaskingForwardOperator(corruption_probability, delta_probability)
    elif corruption_pattern == "averaging":
        assert corruption_probability is not None, "corruption_probability must be specified for averaging corruption pattern"
        assert downsampling_factor is not None, "downsampling_factor must be specified for averaging corruption pattern"
        return AveragingForwardOperator(corruption_probability, downsampling_factor)
    else:
        raise ValueError("Unknown corruption pattern {}".format(corruption_pattern))         


# class AveragingForwardOperator(ForwardOperator):
#     def corrupt(self, x, mask):
#         return x * mask
    
#     def hat_corrupt(self, x, mask):
#         return x / mask


# class Blur(nn.Module):
#     def __init__(self, sigmas, half_size, num_channels=3):
#         super().__init__()
#         self.sigmas = sigmas
#         self.half_size = half_size
#         self.num_channels = num_channels

#     def forward(self, image, t):
#         sigma_weight = t - torch.arange(len(self.sigmas))
#         sigma = self.sigmas[sigma_weight.argmin()]

#         x = torch.arange(-self.half_size, self.half_size + 1).float()
#         kernel_1d = torch.exp(-(x**2) / (2*sigma**2))
#         kernel_1d = kernel_1d / torch.sum(kernel_1d)
        
#         kernel = torch.zeros((self.num_channels, 1, 2*self.half_size+1, 2*self.half_size+1))
#         for i in range(self.num_channels):
#             kernel[i, 0] = torch.outer(kernel_1d, kernel_1d)
                
#         out = F.conv2d(image, kernel.to(image.device), padding=self.half_size, groups=self.num_channels)
#         return out




