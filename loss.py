import torch
import torch.nn.functional as F
from torch import nn

import pytorch_ssim


def criterion(y_true, y_pred, mask, theta=0.1, max_depth_val=1000.0/10.0, device='cpu'):
    if y_true.shape[-1] != y_pred.shape[-1]:
        y_pred = nn.functional.interpolate(
            y_pred, y_true.shape[-2:], mode='bilinear', align_corners=True)
    intr_input = y_pred

    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(1)

    dx_pred, dy_pred = gradient(y_pred)
    dx_true, dy_true = gradient(y_true)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        dx_pred, dy_pred = dx_pred[mask], dy_pred[mask]
        dx_true, dy_true = dx_true[mask], dy_true[mask]
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    l_depth = torch.mean(torch.abs(y_true-y_pred))

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_true - dx_pred))

    ssim_loss = pytorch_ssim.SSIM(device=device)
    l_ssim = torch.clamp(1 - ssim_loss(y_true, y_pred), 0, 1)

    return theta * l_depth + l_edges + l_ssim, intr_input


def gradient(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    return torch.abs(r-l), torch.abs(t-b)


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]




class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


if __name__ == '__main__':
    yt = (torch.randn(4, 1, 348, 1280))
    yp = (torch.randn(4, 1, 348, 1280))
    print(criterion(yp, yt))
