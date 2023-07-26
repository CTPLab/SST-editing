import cv2
import sparse
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
# suppress chained assignment warning
pd.options.mode.chained_assignment = None

cell_comp = {'0': 0, 'Nuclear': 1,
             'Membrane': 2, 'Cytoplasm': 3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test prep_CosMx.py')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/CosMx'),
                        help='Path to NanoString dataset.')
    args = parser.parse_args()
    slide = {'1': 304, '2': 383}

    acc, err = 0, 0
    for s, fov in slide.items():
        pth = args.path / 'GAN' / 'crop' / f'Liver{s}'
        npz = list(pth.rglob('*.npz'))
        print(len(npz))
        # load expr mat dataframe
        df_expr = pd.read_csv(args.path / f'Liver{s}' / 'exprMat_file.csv',
                              index_col=0)
        for n in npz:
            gene = sparse.load_npz(str(n))
            cell = cv2.imread(str(n).replace('_rna.npz', '_cell.png'),
                              flags=cv2.IMREAD_UNCHANGED)
            cell = np.expand_dims(cell, axis=-1)
            cid = n.stem[:-4]
            cnt0 = df_expr.loc[cid].values
            cnt1 = (cell == int(cid.split('_')[-1])) * gene
            cnt1 = cnt1.sum((0, 1)).todense()
            if not (cnt0 == cnt1).all():
                print(cnt0.sum(), cnt1.sum())
                err += 1
            acc += 1
            if acc % 1000 == 0:
                print(acc, err / acc)
        print(acc, err / acc)
