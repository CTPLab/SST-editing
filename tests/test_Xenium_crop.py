
import sys
import cv2
import zarr
import math
import sparse
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow.compute as pc
import pyarrow.parquet as pq

from math import ceil
from pathlib import Path
from random import shuffle
from PIL import Image, ImageFile
from tifffile import imread, imwrite
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from Dataset.utils import _um_to_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


# https://www.biorxiv.org/content/biorxiv/early/2022/11/03/2022.10.06.510405.full.pdf
# First, DAPI images were used to detect nuclei using a neural
# network. Then each nucleus was expanded outwards until either 15 um max distance was
# reached or the boundary of another cell was reached.

# Summary: cv2.fillPoly or cv2.drawContours(..., thickness=-1)
# artifical cell boundary
# there exist cell boundary overlap and nucleus boundary overlap
# nucleus maybe out of cell boundary
# rna expressions maybe out of cell/nucleus boundary (> 8 pixels)

# Conclusion: ~1/10000 rnas have inconsistent assigned cell_id
# in transcripts.parquet and cells.zarr.zip


def test_crop(cell_id, crop_dct, sz,
              df_n, rna_coo, masks):
    for iid, cid in enumerate(cell_id):
        df = df_n[df_n.cell_id == cid][:-1].drop(columns=['cell_id'])
        df = df.mean().astype(int)

        r, c = df.vertex_y, df.vertex_x
        rna_pth = crop_dct[cid]
        crd = rna_pth.stem.split('_')[:-1]

        if r != int(crd[0]) + int(crd[7]) or c != int(crd[1]) + int(crd[9]):
            print(f'coord {r}_{c}')

        rna = sparse.load_npz(str(rna_pth))
        if (rna - rna_coo[r-sz:r+sz, c-sz:c+sz]).nnz != 0:
            print(f'rna {r}_{c}')

        nucl = sparse.load_npz(str(rna_pth).replace('rna.npz',
                                                    'nucl.npz'))
        if (nucl - sparse.COO.from_numpy(masks[0][r-sz:r+sz,
                                                  c-sz:c+sz])).nnz != 0:
            print(f'nucl {r}_{c}')

        cell = sparse.load_npz(str(rna_pth).replace('rna.npz',
                                                    'cell.npz'))
        if (cell - sparse.COO.from_numpy(masks[1][r-sz:r+sz,
                                                  c-sz:c+sz])).nnz != 0:
            print(f'cell {r}_{c}')

        if (iid + 1) % 1000 == 0:
            print(iid, cid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path('Data/Xenium'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--img_type',
                        type=str,
                        choices=('hne', 'dapi'),
                        help='The image type (h&e or Dapi) to be processed')
    parser.add_argument('--roi_size',
                        type=int,
                        default=5000,
                        help='Size of cropped image region.')
    parser.add_argument('--cell_size',
                        type=int,
                        default=128,
                        help='Size of cropped single-cell image.')
    args = parser.parse_args()

    gene_col = ['cell_id', 'y_location', 'x_location', 'feature_name']
    cell_col = ['cell_id', 'y_centroid', 'x_centroid',
                'transcript_counts', 'control_probe_counts',
                'control_codeword_counts', 'total_counts']
    with multiprocessing.Pool(processes=args.core) as pool:
        test_args = list()
        sz = args.cell_size // 2
        for i in ('1', '2'):
            print(i)
            meta_pth = args.root / f'Rep{i}' / 'meta'
            adata = sc.read_10x_h5(filename=str(meta_pth / 'cell_feature_matrix.h5'),
                                   gex_only=False)
            rna_dct = dict(adata.var['feature_types'])
            rna_axs = {k: i for i, k in enumerate(rna_dct)}
            del adata

            crop_pth = args.root / 'GAN' / 'crop' / f'Rep{i}'
            crop_lst = crop_pth.glob('*rna.npz')
            crop_dct = {int(rna.stem.split('_')[2]): rna for rna in crop_lst}
            print(len(crop_dct))

            with zarr.ZipStore(str(meta_pth / 'cells.zarr.zip'), mode='r') as store:
                masks = zarr.group(store=store,  overwrite=False).masks

                print('start transcript')
                rna_coo = sparse.load_npz(str(meta_pth / 'rna.npz'))

                print('start nucleus')
                df_n = pq.read_table(str(meta_pth / 'nucleus_boundaries.parquet'),
                                     columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()
                _um_to_pixel(df_n, 'vertex_{}')
                cell_chunk = np.array_split(np.unique(df_n.cell_id), args.core)
                for chk in cell_chunk:
                    print(chk, '\n')
                    test_args.append([chk, crop_dct, sz,
                                      df_n, rna_coo, masks])
        pool.starmap(test_crop, test_args)
