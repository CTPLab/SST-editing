import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms.functional_tensor as F

from PIL import Image


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, pth, image_filenames, captions, transform=None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.pth = pth
        self.image_filenames = image_filenames
        self.captions = captions
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(str(self.pth / f'{self.image_filenames[idx]}')).convert('RGB')
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous().float()
        if 'Xenium' in str(self.pth):
            img = img[0][None]
        elif 'CosMx' in str(self.pth):
            img = img[1:]
        if img.shape[1] != 128 or img.shape[2] != 128:
            img = F.resize(img, 128, 'bicubic', True).round().clamp(0, 255)

        if self.transform is not None:
            img = self.transform(img)
        img = img / 127.5 - 1 
        return {'image': img, 
                'caption': self.captions[idx].astype(np.float32)}

    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train", size=224):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    