import cv2
import sparse
import tiledb
import argparse
import tiledbsoma
import multiprocessing

import numpy as np
import pandas as pd
from pathlib import Path
# suppress chained assignment warning
pd.options.mode.chained_assignment = None

cell_comp = {'0': 0, 'Nuclear': 1,
             'Membrane': 2, 'Cytoplasm': 3}


def test_fov(dfs, pth, fov, meta_cell):
    prefix = f'{fov}_'
    df = dfs[dfs.index.str.startswith(prefix)]
    # load dataframe processed sparse spatial data
    rna_pth = pth / 'GeneLabels' / f'GeneLabels_F{str(fov).zfill(3)}.npz'
    if not rna_pth.is_file():
        print(rna_pth, len(df), 'empty')
        return

    if meta_cell:
        cell_pth = pth / 'CellLabels' / f'CellLabels_F{str(fov).zfill(3)}.tif'
        cell_img = cv2.imread(str(cell_pth), flags=cv2.IMREAD_UNCHANGED)
    else:
        # test compartment label
        comp_pth = pth / 'GeneLabels' / \
            f'CompLabels_F{str(fov).zfill(3)}.npz'
        comp_np = sparse.load_npz(str(comp_pth)).todense()

        comp_pth = pth / 'CompartmentLabels' / \
            f'CompartmentLabels_F{str(fov).zfill(3)}.tif'
        comp_im = cv2.imread(str(comp_pth), flags=cv2.IMREAD_UNCHANGED)

        assert np.all(comp_np[comp_np != 0] == comp_im[comp_np != 0]), \
            f'{rna_pth}: compartment label does not fully overlap.'
        del comp_np, comp_im

        cell_pth = pth / 'GeneLabels' / f'CellLabels_F{str(fov).zfill(3)}.npz'
        cell_img = sparse.load_npz(str(cell_pth)).todense()
    cell_img = np.expand_dims(cell_img, axis=-1)

    # test transcript count
    rna_coo = sparse.load_npz(str(rna_pth))
    df.index = df.index.map(lambda x: x[len(prefix):]).astype(int)
    for cid in df.index:
        cnt = rna_coo * (cell_img == cid)
        if not (cnt.sum((0, 1)).todense() == df.loc[cid].values).all():
            print(fov, cid)
    print(rna_pth, len(df), 'done')


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
    parser.add_argument('--meta_cell',
                        action='store_true',
                        help='Whether to visualize the entire processed labels.')
    args = parser.parse_args()
    slide = {'1': 304, '2': 383}

    with multiprocessing.Pool(processes=args.core) as pool:
        test_args = list()
        for s, fov in slide.items():
            pth = args.path / f'Liver{s}'
            # load expr mat dataframe
            df_expr = pd.read_csv(pth / 'exprMat_file.csv', index_col=0)
            df_expr.index = df_expr.index.map(lambda x: x[4:])
            for f in range(1, fov + 1):
                test_args.append((df_expr, pth, f, args.meta_cell))
        pool.starmap(test_fov, test_args)
