import torch
import random
import sparse
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def random_rotation(x: torch.Tensor) -> torch.Tensor:
    x = torch.rot90(x, random.randint(0, 3), [1, 2])
    return x


transform = transforms.Compose(
    [transforms.Lambda(lambda x: random_rotation(x)),
     transforms.RandomHorizontalFlip()]
)


def random_rotation_sp(x):
    # img, gene = x
    t = random.randint(0, 3)
    if t > 0:
        x[0] = torch.rot90(x[0], t, [1, 2])
        # gene has HWC format (for spconv),
        # thus transpose HW axis (0,1 dim)
        for _ in range(t):
            x[1] = x[1].transpose((1, 0, 2))
            # reverse the H axis
            x[1].coords[0] = x[1].shape[0] - 1 - x[1].coords[0]
    return x


def random_hflip_sp(x, p=0.5):
    # img, gene = x
    if torch.rand(1) < p:
        x[0] = F.hflip(x[0])
        # gene has HWC format (for spconv),
        # reverse the W axis
        x[1].coords[1] = x[1].shape[1] - 1 - x[1].coords[1]
    return x


transform_sp = transforms.Compose(
    [transforms.Lambda(lambda x : random_rotation_sp(x)),
     transforms.Lambda(lambda x: random_hflip_sp(x))]
)

if __name__ == '__main__':
    gene_pth = list(Path('Data/CosMx/GAN/crop').rglob('*_rna.npz'))
    random.shuffle(gene_pth)
    print('shuffle done!')
    for pid, pth in enumerate(gene_pth):
        gene_sp = sparse.load_npz(str(pth))
        gene_ts = torch.from_numpy(gene_sp.todense().transpose(2, 0, 1))
        gene_ts, gene_sp = random_rotation_sp([gene_ts, gene_sp])
        gene_ts, gene_sp = random_hflip_sp([gene_ts, gene_sp])
        g_sp = sparse.COO.from_numpy(gene_ts.numpy())
        gene_sp = gene_sp.transpose((2, 0, 1))
        assert (g_sp - gene_sp).nnz == 0
        assert (g_sp.todense() == gene_sp.todense()).all()
        if pid % 50 == 0:
            print(pid)
        if pid == 1000:
            break
