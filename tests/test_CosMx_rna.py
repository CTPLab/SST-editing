import cv2
import sparse
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
# suppress chained assignment warning
pd.options.mode.chained_assignment = None

cell_comp = {'0': 0, 'Nuclear': 1,
             'Membrane': 2, 'Cytoplasm': 3}


def test_assign(df, fov, cell_fld, expr_dct, vis=False, hei=3648):
    # load dataframe processed sparse spatial data
    rna_pth = cell_fld / 'GeneLabels' / \
        f'GeneLabels_F{str(fov).zfill(3)}.npz'
    rna_coo = sparse.load_npz(str(rna_pth))

    comp_pth = cell_fld / 'GeneLabels' / \
        f'CompLabels_F{str(fov).zfill(3)}.npz'
    comp_coo = sparse.load_npz(str(comp_pth))

    cell_pth = cell_fld / 'GeneLabels' / \
        f'CellLabels_F{str(fov).zfill(3)}.npz'
    cell_coo = sparse.load_npz(str(cell_pth))

    region = str(cell_fld).split('/')[2]
    dat_info = f'{region}-{fov}'
    print(dat_info)
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
        vis_file = cell_fld / 'Visual' / f'F{str(fov).zfill(3)}.png'
        cv2.imwrite(str(vis_file), vis_np.astype(np.uint16))
    else:
        # preprocess the dataframe
        df = df[['y_local_px', 'x_local_px', 'target', 'cell_ID', 'CellComp']]
        df.loc[df.y_local_px == '254.3598632812µ', 'y_local_px'] = '254.3598'
        # use cop y() to avoid 'chained assignment' warning
        df.CellComp = df.CellComp.replace(cell_comp)
        df.target = df.target.replace(expr_dct)
        df.y_local_px = hei - df.y_local_px.astype(float).round()
        df.x_local_px = df.x_local_px.astype(float).round()

        # check if there are any different cell_ID or
        # CellComp values for overlapped rnas with same pos
        grp = ['y_local_px', 'x_local_px', 'target']
        comp_unq = df.drop(columns=['cell_ID']).groupby(grp).CellComp.nunique()
        cid_unq = df.drop(columns=['CellComp']).groupby(grp).cell_ID.nunique()
        assert np.all(comp_unq.values == 1) and np.all(cid_unq.values == 1), \
            f'{dat_info}: cell_ID or CellComp not unique.'

        # assign rna, cellcomp and cellid values
        # to the their spatial pos
        expr = zip(df.y_local_px, df.x_local_px,
                   df.cell_ID, df.target, df.CellComp)
        rna_np = np.zeros(rna_coo.shape, dtype=int)
        comp_np = np.zeros(comp_coo.shape, dtype=int)
        cell_np = np.zeros(cell_coo.shape, dtype=int)
        for r, c, cell, tar, comp in expr:
            r, c = int(r), int(c)
            rna_np[r, c, tar] += 1
            # as the overlapped cellcomp and cellid are identical,
            # there are no changes for repetitive assignment
            comp_np[r, c] = comp
            cell_np[r, c] = cell

        # compared to the dataframe processed spatial rna data
        rna_dif = (rna_coo - sparse.COO.from_numpy(rna_np)).nnz
        assert rna_dif == 0, f'{dat_info}: rna {rna_dif} not consistent.'
        comp_dif = (comp_coo - sparse.COO.from_numpy(comp_np)).nnz
        assert comp_dif == 0, f'{dat_info}: comp {comp_dif} not consistent.'
        cell_dif = (cell_coo - sparse.COO.from_numpy(cell_np)).nnz
        assert cell_dif == 0, f'{dat_info}: cell {cell_dif} not consistent.'


def test_mask(fov, cell_fld):
    comp_pth = cell_fld / 'GeneLabels' / \
        f'CompLabels_F{str(fov).zfill(3)}.npz'
    comp_np = sparse.load_npz(str(comp_pth)).todense()

    cell_pth = cell_fld / 'GeneLabels' / \
        f'CellLabels_F{str(fov).zfill(3)}.npz'
    cell_np = sparse.load_npz(str(cell_pth)).todense()

    comp_pth = cell_fld / 'CompartmentLabels' / \
        f'CompartmentLabels_F{str(fov).zfill(3)}.tif'
    comp_im = cv2.imread(str(comp_pth), flags=cv2.IMREAD_UNCHANGED)

    cell_pth = cell_fld / 'CellLabels' / \
        f'CellLabels_F{str(fov).zfill(3)}.tif'
    cell_im = cv2.imread(str(cell_pth), flags=cv2.IMREAD_UNCHANGED)

    data_info = str(cell_fld) + f'_{fov}'
    assert np.all(comp_np[comp_np != 0] == comp_im[comp_np != 0]), \
        f'{data_info}: compartment label does not fully overlap.'
    assert np.all(cell_np[cell_np != 0] == cell_im[cell_np != 0]), \
        f'{data_info}: cell label does not fully overlap.'


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
        gen_args = list()
        for fld in args.path.iterdir():
            if fld.is_dir() and 'Lung' in str(fld):
                cell_fld = fld / f'{fld.name}-Flat_files_and_images'
                if args.test_assign:
                    df_expr = pd.read_csv(cell_fld / f'{fld.name}_exprMat_file.csv',
                                          encoding='ISO-8859-1')

                    cell_id = df_expr.columns[0:2].tolist()
                    expr_nm = df_expr.columns.difference(
                        cell_id, sort=False).tolist()
                    expr_dct = {e: i for i, e in enumerate(expr_nm)}

                    df_tx = pd.read_csv(cell_fld / f'{fld.name}_tx_file.csv',
                                        encoding='ISO-8859-1', low_memory=False)
                    if fld.name == 'Lung5_Rep2':
                        cnd = (df_tx.CellComp.notnull()) & \
                            (df_tx.CellComp != 'Cytoplasí') & \
                            (df_tx.target != 'Cytoplasm')
                        df_tx = df_tx[cnd]

                    if args.visual:
                        (cell_fld / 'Visual').mkdir(parents=True, exist_ok=True)

                    print(fld.name)
                    fov = np.unique(df_tx.fov)
                    for f in fov:
                        df = df_tx[df_tx.fov == f]
                        gen_args.append((df, f, cell_fld,
                                         expr_dct, args.visual))
                if args.test_mask:
                    gene_lst = (cell_fld / 'GeneLabels').\
                        glob('GeneLabels*.npz')
                    for fov in gene_lst:
                        f = int(fov.stem[-3:])
                        print(fov, f)
                        gen_args.append((f, cell_fld))
        if args.test_assign:
            pool.starmap(test_assign, gen_args)
        if args.test_mask:
            pool.starmap(test_mask, gen_args)
