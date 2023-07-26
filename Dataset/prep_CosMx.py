import sys
import cv2
import random
import sparse
import tiledb
import argparse
import tiledbsoma

import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path
from random import shuffle
from tifffile import imread
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


def gen_crop(df, fov,
             sz, fov_sz,
             pth, fov_pth,
             debug=False,
             prefx='GeneLabels',
             num=5):
    # rna expr image show boundary effects
    # e.g., minor missing gene expr for row=64, ..., 70
    # Compartment label can be tricky,
    # e.g., Membrane represent non-smooth, twisted small regions,
    # which may not be useful.
    rna_pth = fov_pth / f'{prefx}' / f'{prefx}_F{str(fov).zfill(3)}'
    rna_coo = sparse.load_npz(str(rna_pth) + '.npz')
    flo_pth = str(rna_pth).replace(prefx, 'CellComposite')
    flo_img = cv2.imread(flo_pth + '.jpg', flags=cv2.IMREAD_UNCHANGED)
    comp_pth = str(rna_pth).replace(prefx, 'CompartmentLabels')
    comp_img = cv2.imread(comp_pth + '.tif', flags=cv2.IMREAD_UNCHANGED)
    cell_pth = str(rna_pth).replace(prefx, 'CellLabels')
    cell_img = cv2.imread(cell_pth + '.tif', flags=cv2.IMREAD_UNCHANGED)

    df_sub = zip(df.index, df.y_FOV_px, df.x_FOV_px)
    if debug:
        df_sub = list(df_sub)
        shuffle(df_sub)
        # prep the raw cell mask which has minor shift
        mask_pth = str(rna_pth).replace(prefx, 'CellOverlay')
        mask = cv2.imread(mask_pth + '.jpg', flags=cv2.IMREAD_UNCHANGED)
        # improve the visualization
        rna_coo *= 20000
        comp_img *= 50
        cell_img *= 50
        cell_img += np.sign(cell_img) * 20000

    acc, sz = 0, sz // 2
    for _, (cid, r, c) in enumerate(df_sub):
        if sz + 512 <= r <= fov_sz - sz and \
           sz + 512 <= c <= fov_sz - sz:
            rna = rna_coo[r - sz: r + sz, c - sz: c + sz]
            flo = flo_img[r - sz: r + sz, c - sz: c + sz]
            comp_lab = comp_img[r - sz: r + sz, c - sz: c + sz]
            cell_lab = cell_img[r - sz: r + sz, c - sz: c + sz]
            if debug:
                cv2.imwrite(str(pth / f'{cid}_tot.png'),
                            rna.sum(axis=-1).todense().astype(np.uint16))
                cv2.imwrite(str(pth / f'{cid}_flo.jpg'), flo)
                mk = mask[r - sz: r + sz, c - sz: c + sz].copy()
                cv2.circle(mk, (sz, sz), radius=2,
                           color=(0, 0, 255), thickness=2)
                cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mk)

                cp = comp_lab.copy()
                cl = cell_lab.copy().astype(np.uint16)
                cv2.imwrite(str(pth / f'{cid}_comp.png'), cp)
                cv2.imwrite(str(pth / f'{cid}_cell.png'), cl)

                cv2.circle(mask, (c,r), radius=2, color=(0, 0, 255), thickness=4)
                acc += 1
                if acc >= num:
                    # cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mask)
                    break
            else:
                sparse.save_npz(str(pth / f'{cid}_rna'), rna)
                cv2.imwrite(str(pth / f'{cid}_flo.png'), flo)
                cv2.imwrite(str(pth / f'{cid}_comp.png'), comp_lab)
                cv2.imwrite(str(pth / f'{cid}_cell.png'),
                            cell_lab.astype(np.uint16))
    print(rna_pth, len(df), 'done')


def prep_gene(trans, qc, sz,
              pth, fov,
              expr_dct,
              cell_comp):
    df = trans.query(cond=qc).df[:]
    if df.empty:
        print(qc, 'empty')
        return

    df = df.drop(columns=['slideID', 'fov', 'z_FOV_slice', 'cell_id'])
    df.target = df.target.replace(expr_dct)
    df.CellComp = df.CellComp.replace(cell_comp)

    len1 = len(df.drop_duplicates())
    len2 = len(df.drop(columns=['CellComp', 'CellId']).drop_duplicates())
    df = df.groupby(list(df.columns), as_index=False).size()
    print(qc, len1 - len2, len1 - len(df), len(df[df['CellComp'] == 0]))

    y = df.y_FOV_px.values
    x = df.x_FOV_px.values
    gene_coo = sparse.COO((y, x, df.target.values),
                          df['size'].values,
                          shape=[sz, sz, len(expr_dct)])
    sparse.save_npz(str(pth / f'GeneLabels_F{str(fov).zfill(3)}'),
                    gene_coo)

    df = df.drop(columns=['target', 'size']).drop_duplicates()
    y = df.y_FOV_px.values
    x = df.x_FOV_px.values

    comp_coo = sparse.COO((y, x), df.CellComp.values, shape=[sz, sz])
    sparse.save_npz(str(pth / f'CompLabels_F{str(fov).zfill(3)}'),
                    comp_coo)

    cell_coo = sparse.COO((y, x), df.CellId.values, shape=[sz, sz])
    sparse.save_npz(str(pth / f'CellLabels_F{str(fov).zfill(3)}'),
                    cell_coo)


def prep_meta(path, slide, pySoma):
    # prepare summary count table for each cell
    if not (path / 'Liver1' / 'exprMat_file.csv').is_file():
        counts = pySoma['RNA'].X['counts'].df().reset_index(
            level=['obs_id', 'var_id'])
        for s, fov in slide.items():
            cnt = counts[counts.obs_id.str.startswith(f'c_{s}')]
            cnt = cnt.pivot(index='obs_id', columns='var_id',
                            values='value')
            # remove the index name obs_id and column name var_id
            cnt = cnt.rename_axis(None, axis=1).rename_axis(None, axis=0)
            cnt = cnt.fillna(0).astype(int)
            csv_pth = path / f'Liver{s}' / 'exprMat_file.csv'
            cnt.to_csv(str(csv_pth))
            print(s, fov, len(cnt))
            print(cnt.head())


def prep_img(img_pth, save_pth, chns, 
             extm, bdry):
    img = imread(str(img_pth))[chns]
    img = np.clip(img.astype(np.float32),
                  a_min=extm[:, 0][:, None, None],
                  a_max=extm[:, 1][:, None, None])
    img = img.transpose((1, 2, 0))
    # imin = img.min(axis=(0, 1))
    # imax = img.max(axis=(0, 1))
    img = (img - bdry[:, 0]) / (bdry[:, 1] - bdry[:, 0])
    # print(img_pth.stem, imin, imax,
    #       img.min(axis=(0, 1)), img.max(axis=(0, 1)))
    img = (img * 255).astype(np.uint8)
    img_nm = img_pth.stem.split('_')[-1]
    i0 = np.zeros_like(img[:, :, 0])[:, :, None]
    img = np.concatenate((i0, img), -1)
    Image.fromarray(img).save(str(save_pth / f'CellComposite_{img_nm}.jpg'))
    # for cid, c in enumerate(chns):
    #     Image.fromarray(img[:,:,cid]).save(str(save_pth / f'CellImage_{img_nm}_{c}.jpg'))


def gamma_trans(gamma):
    gtab = [np.power(x / 255, gamma) * 255 for x in range(256)]
    gtab = np.array(gtab).astype(np.uint8)
    return gtab

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/CosMx'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--fov_sz',
                        type=int,
                        default=4256,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--crop_sz',
                        type=int,
                        default=160,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--prep_img',
                        action='store_true',
                        help='Prepare the entire gene (rna) expr image.')
    parser.add_argument('--prep_meta',
                        action='store_true',
                        help='Prepare the entire gene (rna) expr image.')
    parser.add_argument('--prep_gene',
                        action='store_true',
                        help='Prepare the entire gene (rna) expr image.')
    parser.add_argument('--prep_crop',
                        action='store_true',
                        help='Prepare the cropped pair of cell and gene images.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to visualize additional cell mask.')
    args = parser.parse_args()

    # CD298/B2M
    # 23.0 91.0  631.0  Liver1
    # 90.0 379.0 1550.0 Liver2
    # Dapi
    # 12.0 92.0  1842.0 Liver1
    # 24.0 387.0 2375.0 Liver2
    if args.prep_img:
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args, chns = list(), [2, 4] 
            bdry = np.array([[23, 1550], [12, 2375]])
            for i in ('Liver1', 'Liver2'):
                extm = [[23, 631], [12, 1842]] if i == 'Liver1' else [[90, 1550], [24, 2375]]
                extm = np.array(extm)
                data_pth = args.path / i
                save_pth = data_pth / 'CellComposite'
                save_pth.mkdir(parents=True, exist_ok=True)
                img_pth = list((data_pth / 'Morphology2D_Normalized').glob('*.TIF'))
                img_pth.sort()
                for p in img_pth:
                    prep_args.append((p, save_pth, chns, extm, bdry))
            pool.starmap(prep_img, prep_args)


    csv_file = ('tx', 'exprMat', 'metadata')
    img_type = (('CellComposite', 'jpg'),
                ('CellLabels', 'tif'),
                ('CellOverlay', 'jpg'),
                ('CompartmentLabels', 'tif'),
                # GeneLabels need to be processed
                ('GeneLabels', 'npz'))
    cell_comp = {'0': 0, 'Nuclear': 1,
                 'Membrane': 2, 'Cytoplasm': 3}
    slide = {'1': 304, '2': 383}

    # this implementation is version sensitive
    # tiledb==0.20.0
    # tiledbsoma==0.1.22
    config = tiledb.Config()
    ctx = tiledb.Ctx(config)
    pySoma = tiledbsoma.SOMACollection(str(args.path / 'raw' / 'LiverDataRelease'),
                                       ctx=ctx)

    if args.prep_meta:
        prep_meta(args.path, slide, pySoma)

    if args.prep_gene:
        # Prepare the entire gene expr images
        with multiprocessing.Pool(processes=args.core) as pool:
            gen_args = list()
            trans = tiledb.open(pySoma['RNA'].obsm['transcriptCoords'].uri,
                                'r', ctx=ctx)
            for s, fov in slide.items():
                pth = args.path / f'Liver{s}'
                gene_pth = pth / img_type[-1][0]
                gene_pth.mkdir(parents=True, exist_ok=True)
                # load expr mat dataframe
                df_expr = pd.read_csv(pth / 'exprMat_file.csv', index_col=0)
                expr_dct = {e: i for i, e in enumerate(df_expr.columns)}

                for f in range(1, fov + 1):
                    qc = f'slideID == {s} and fov == {f}'
                    gen_args.append((trans, qc, args.fov_sz,
                                     gene_pth, f,
                                     expr_dct, cell_comp))
            pool.starmap(prep_gene, gen_args)

    if args.prep_crop:
        # Prepare single-cell images
        with multiprocessing.Pool(processes=args.core) as pool:
            gen_args = list()
            df_cell = pySoma['RNA'].obs.df(attrs=['y_FOV_px', 'x_FOV_px'])
            df_cell = df_cell.rename_axis(None, axis=0)
            for s, fov in slide.items():
                pth = args.path / f'Liver{s}'
                if args.debug:
                    crop_pth = pth / 'debug_crop'
                else:
                    crop_pth = pth.parent / 'GAN' / 'crop' / pth.name
                crop_pth.mkdir(parents=True, exist_ok=True)
                print(crop_pth)

                sld = f'c_{s}'
                dfs = df_cell[df_cell.index.str.startswith(sld)]
                for f in range(1, fov + 1):
                    sld_fov = f'{sld}_{f}_'
                    df = dfs[dfs.index.str.startswith(sld_fov)]
                    if df.empty:
                        npz_pth = pth / 'GeneLabels' / \
                            f'GeneLabels_F{str(f).zfill(3)}.npz'
                        print(sld_fov, npz_pth.is_file(), 'empty')
                    else:
                        gen_args.append((df, f,
                                         args.crop_sz, args.fov_sz,
                                         crop_pth, pth,
                                         args.debug))
            pool.starmap(gen_crop, gen_args)
