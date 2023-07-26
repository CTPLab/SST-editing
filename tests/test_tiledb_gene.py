
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


def test_assign(trans, qc,
                fov, pth, expr_dct,
                vis=False):
    df = trans.query(cond=qc).df[:]
    if df.empty:
        print(qc, 'empty')
        return

    # load dataframe processed sparse spatial data
    rna_pth = pth / f'GeneLabels_F{str(fov).zfill(3)}.npz'
    rna_coo = sparse.load_npz(str(rna_pth))

    comp_pth = pth / f'CompLabels_F{str(fov).zfill(3)}.npz'
    comp_coo = sparse.load_npz(str(comp_pth))

    cell_pth = pth / f'CellLabels_F{str(fov).zfill(3)}.npz'
    cell_coo = sparse.load_npz(str(cell_pth))
    print(qc)
    if vis:
        rna_coo *= 20000
        rna_np = rna_coo.sum(axis=-1).todense()

        comp_coo *= 10000
        comp_coo += np.sign(comp_coo) * 10000
        comp_np = comp_coo.todense()

        cell_coo *= 50
        cell_coo += np.sign(cell_coo) * 20000
        cell_np = cell_coo.todense()
        vis_np = np.stack((rna_np, comp_np, cell_np), axis=-1)
        vis_file = pth.parent / 'Visual' / f'F{str(fov).zfill(3)}.png'
        cv2.imwrite(str(vis_file), vis_np.astype(np.uint16))
    else:
        # preprocess the dataframe
        df = df[['y_FOV_px', 'x_FOV_px', 'target', 'CellId', 'CellComp']]
        df.CellComp = df.CellComp.replace(cell_comp)
        df.target = df.target.replace(expr_dct)

        # check if there are any different cell_ID or
        # CellComp values for overlapped rnas with same pos
        grp = ['y_FOV_px', 'x_FOV_px', 'target']
        comp_unq = df.drop(columns=['CellId']).groupby(grp).CellComp.nunique()
        cid_unq = df.drop(columns=['CellComp']).groupby(grp).CellId.nunique()
        assert np.all(comp_unq.values == 1) and np.all(cid_unq.values == 1), \
            f'{qc}: cell_ID or CellComp not unique.'

        # assign rna, cellcomp and cellid values
        # to the their spatial pos
        expr = zip(df.y_FOV_px, df.x_FOV_px,
                   df.target, df.CellId, df.CellComp)
        rna_np = np.zeros(rna_coo.shape, dtype=np.uint8)
        comp_np = np.zeros(comp_coo.shape, dtype=np.uint8)
        cell_np = np.zeros(cell_coo.shape, dtype=int)
        for r, c, t, cell, comp in expr:
            r, c = int(r), int(c)
            rna_np[r, c, t] += 1
            # as the overlapped cellcomp and cellid are identical,
            # there are no changes for repetitive assignment
            comp_np[r, c] = comp
            cell_np[r, c] = cell

        # compared to the dataframe processed spatial rna data
        rna_dif = (rna_coo - sparse.COO.from_numpy(rna_np)).nnz
        assert rna_dif == 0, f'{qc}: rna {rna_dif} not consistent.'
        comp_dif = (comp_coo - sparse.COO.from_numpy(comp_np)).nnz
        assert comp_dif == 0, f'{qc}: comp {comp_dif} not consistent.'
        cell_dif = (cell_coo - sparse.COO.from_numpy(cell_np)).nnz
        assert cell_dif == 0, f'{qc}: cell {cell_dif} not consistent.'


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
    parser.add_argument('--visual',
                        action='store_true',
                        help='Whether to visualize the entire processed labels.')
    args = parser.parse_args()
    # this implementation is version sensitive
    # tiledb==0.20.0
    # tiledbsoma==0.1.22
    config = tiledb.Config()
    ctx = tiledb.Ctx(config)
    pySoma = tiledbsoma.SOMACollection(str(args.path / 'raw' / 'LiverDataRelease'),
                                       ctx=ctx)
    slide = {'1': 304, '2': 383}

    with multiprocessing.Pool(processes=args.core) as pool:
        test_args = list()
        trans = tiledb.open(pySoma['RNA'].obsm['transcriptCoords'].uri,
                            'r', ctx=ctx)
        for s, fov in slide.items():
            pth = args.path / f'Liver{s}'
            # load expr mat dataframe
            df_expr = pd.read_csv(pth / 'exprMat_file.csv', index_col=0)
            expr_dct = {e: i for i, e in enumerate(df_expr.columns)}

            if args.visual:
                (pth / 'Visual').mkdir(parents=True, exist_ok=True)

            for f in range(1, fov + 1):
                qc = f'slideID == {s} and fov == {f}'
                test_args.append((trans, qc, f,
                                  pth / 'GeneLabels',
                                  expr_dct, args.visual))
        pool.starmap(test_assign, test_args)
