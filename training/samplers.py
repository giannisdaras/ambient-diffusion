import torch

def backward_sde_sampler(net, x_hat, class_labels, t_cur, t_next, augment_labels, **kwargs):
    denoised = net(x_hat, t_cur, class_labels, augment_labels=augment_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_cur
    x_next = x_hat + 2 * (t_next - t_cur) * d_cur  + torch.sqrt(2 * (t_cur - t_next).abs() * t_cur) * torch.randn_like(x_hat)
    return x_next, denoised