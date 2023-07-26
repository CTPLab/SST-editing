import torch
import numpy as np 
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
# import torchvision.transforms.functional_tensor as F


resize = transforms.Resize(128,
                           interpolation=transforms.InterpolationMode.BICUBIC,
                           antialias=True)

chn, sz = 3, 160
a = (np.random.rand(chn, sz, sz) * 255).astype(np.uint8).astype(np.float32)
if chn == 3:
    a[0] = 0
a = torch.from_numpy(a).contiguous()
print(a.shape)
b = resize(a)
b1 = resize(a[1:])
b2 = transforms.functional_tensor.resize(a, 128, interpolation='bicubic', antialias=True)
print((b[1:] == b1).all())
print((b[0] == 0).all())
print((b == b2).all())

print(a[0].shape, a[0][None].shape)
