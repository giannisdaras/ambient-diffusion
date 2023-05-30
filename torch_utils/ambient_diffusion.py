import torch
import numpy as np


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