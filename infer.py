# MIT License
import matplotlib
# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import numpy as np
from torch import nn
from torchvision.transforms import ToTensor
from PIL import Image

# from train_mono import load_ckpt
# from zoedepth.utils.misc import get_image_from_url, colorize
import torch

from Initial_stage.models.HidingUNet import UnetGenerator
from model import DDNet
from torchvision.transforms import functional as F

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

# print("*" * 20 + " Testing zoedepth " + "*" * 20)
# conf = get_config("zoedepth", "infer")


print("Config:")
# pprint(conf)
path = "data-x/LAPTOP-33576NBL__2024-12-14-20_54_01/checkPoints/Hnet_epoch_0,sumloss=0.970695,ssimloss=0.191964.pth"
model = DDNet()
model.load_state_dict(torch.load('data-x/LAPTOP-33576NBL__2024-12-14-02_28_13/checkPoints/netD_epoch_0,sumloss=0.495543,mseloss=0.056215.pth'))
# model =  UnetGenerator(input_nc=3, output_nc=1, num_downs= 7,  output_function=nn.Sigmoid)
# torch.load(path)
model.to(DEVICE)
model.eval()
x = torch.rand(1, 3, 384, 512).to(DEVICE)

print("-"*20 + "Testing on a random input" + "-"*20)

with torch.no_grad():
    out = model(x)

if isinstance(out, dict):
    # print shapes of all outputs
    for k, v in out.items():
        if v is not None:
            print(k, v.shape)
else:
    print([o.shape for o in out if o is not None])

print("\n\n")
print("-"*20 + " Testing on an indoor scene from url " + "-"*20)

# Test img
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"
# img = get_image_from_url(url)
img = Image.open('image.png')
orig_size = img.size
X = ToTensor()(img)
X = X.unsqueeze(0).to(DEVICE)
shape = X.shape
print("X.shape", X.shape)
print("predicting")
X = F.resize(X, [384, 384])
with torch.no_grad():
    out = model(X).cpu()
# depth = np.load("image_depth.png").squeeze()
# or just,
# out = model.infer_pil(img)
out = y_pred = nn.functional.interpolate(
            out, shape[-2:], mode='bilinear', align_corners=True)

print("output.shape", out.shape)
pred = Image.fromarray(colorize(out))
# Stack img and pred side by side for comparison and save
pred = pred.resize(orig_size, Image.Resampling.LANCZOS)
stacked = Image.new("RGB", (orig_size[0]*2, orig_size[1]))
stacked.paste(img, (0, 0))
stacked.paste(pred, (orig_size[0], 0))

stacked.save("pred2.png")
print("saved pred.png")


# model.infer_pil(img, output_type="pil").save("pred_raw.png")