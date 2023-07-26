import cv2
import sparse
import argparse
import multiprocessing

import numpy as np
from pathlib import Path


def test_stat(pth):
    comp_np = sparse.load_npz(str(pth)).todense()
    comp_im = cv2.imread(str(pth).replace('.npz', '.png'),
                         flags=cv2.IMREAD_UNCHANGED)
    assert np.all(comp_np[comp_np != 0] == comp_im[comp_np != 0]), \
        f'{pth.name}: comp inconsistent.'

    cell = str(pth).replace('comp.npz', 'cell.npz')
    cell_np = sparse.load_npz(cell).todense()
    cell_im = cv2.imread(cell.replace('.npz', '.png'),
                         flags=cv2.IMREAD_UNCHANGED)

    assert np.all(cell_np[cell_np != 0] == cell_im[cell_np != 0]), \
        f'{pth.name}: cell inconsistent.'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test prep_CosMx.py')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/CosMx'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--test_assign',
                        action='store_true',
                        help='Test the gene spatial assign with a different equivalent algorithm')
    parser.add_argument('--test_mask',
                        action='store_true',
                        help='Test the assign wih the gt cell and comparment label images')
    parser.add_argument('--visual',
                        action='store_true',
                        help='Whether to visualize the entire processed labels.')
    args = parser.parse_args()
    with multiprocessing.Pool(processes=args.core) as pool:
        test_args = list()
        for fld in args.path.iterdir():
            if fld.is_dir() and 'Lung' in str(fld):
                print(fld)
                crop_fld = fld.parent / 'GAN' / 'crop' / fld.name
                crop = crop_fld.glob('*comp.npz')
                for c in crop:
                    test_args.append([c])
        pool.starmap(test_stat, test_args)
