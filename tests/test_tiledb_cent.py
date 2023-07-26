import sys
import cv2
import sparse
import tiledb
import argparse
import tiledbsoma

import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path
from random import shuffle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


def test_cent(df, fov,
              sz, fov_sz,
              pth, fov_pth,
              prefx='GeneLabels',
              num=10):
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
    mask_pth = str(rna_pth).replace(prefx, 'CellOverlay')
    mask = cv2.imread(mask_pth + '.jpg', flags=cv2.IMREAD_UNCHANGED)

    df_sub = zip(df.index, df.y_FOV_px, df.x_FOV_px)
    df_sub = list(df_sub)
    # # improve the visualization
    # rna_coo *= 20000
    # comp_img *= 50
    # cell_img *= 50
    # cell_img += np.sign(cell_img) * 20000

    acc, sz = 0, sz // 2
    for rid, (cid, r, c) in enumerate(df_sub):
        if sz <= r <= fov_sz - sz and \
           sz <= c <= fov_sz - sz:
            lr_s, rr_s = max(0, r - sz), min(fov_sz, r + sz)
            lc_s, rc_s = max(0, c - sz), min(fov_sz, c + sz)
            lr_l, rr_l = max(0, r - 2 * sz), min(fov_sz, r + 2 * sz)
            lc_l, rc_l = max(0, c - 2 * sz), min(fov_sz, c + 2 * sz)
            cell_s = cell_img[lr_s: rr_s, lc_s: rc_s]
            cell_l = cell_img[lr_l: rr_l, lc_l: rc_l]
            comp_s = comp_img[lr_s: rr_s, lc_s: rc_s]
            comp_l = comp_img[lr_l: rr_l, lc_l: rc_l]

            cnum = int(cid.split('_')[-1])
            mask_s = ((cell_s == cnum) & (comp_s == 1))
            maks_l = ((cell_l == cnum) & (comp_l == 1))
            sum_s, sum_l = mask_s.sum(), maks_l.sum()
            assert sum_l >= sum_s
            
            if sum_s != 0:
                crd = np.nonzero(mask_s)
                loc_r = int(crd[0].mean())
                loc_c = int(crd[1].mean())
                r, c = lr_s + loc_r, lc_s + loc_c

            # if (sum_s != sum_l) or sum_l == 0:
            #     if rid >= 500:
            #         continue
            if (sum_s != sum_l):
                flo = flo_img[r - sz: r + sz, c - sz: c + sz]
                cv2.imwrite(str(pth / f'{cid}_flo.jpg'), flo)
                mk = mask[r - sz: r + sz, c - sz: c + sz].copy()
                cv2.circle(mk, (sz, sz), radius=2,
                           color=(0, 0, 255), thickness=2)
                cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mk)
                cm = comp_img[r - sz: r + sz, c - sz: c + sz]
                cv2.imwrite(str(pth / f'{cid}_cmp.jpg'), cm * 50)

            if (sum_s != sum_l):
                print(cid, sum_s, sum_l)
            elif sum_s == 0:
                acc += 1

        #     rna = rna_coo[r - sz: r + sz, c - sz: c + sz]
        #     flo = flo_img[r - sz: r + sz, c - sz: c + sz]
        #     comp_lab = comp_img[r - sz: r + sz, c - sz: c + sz]
        #     cell_lab = cell_img[r - sz: r + sz, c - sz: c + sz]

        # if rid < num:
        #     cv2.imwrite(str(pth / f'{cid}_tot.png'),
        #                 rna.sum(axis=-1).todense().astype(np.uint16))
        #     cv2.imwrite(str(pth / f'{cid}_flo.jpg'), flo)
        #     mk = mask[r - sz: r + sz, c - sz: c + sz].copy()
        #     cv2.circle(mk, (sz, sz), radius=2,
        #                 color=(0, 0, 255), thickness=2)
        #     cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mk)

            # cp = comp_lab.copy()
            # cl = cell_lab.copy().astype(np.uint16)
            # cv2.imwrite(str(pth / f'{cid}_comp.png'), cp)
            # cv2.imwrite(str(pth / f'{cid}_cell.png'), cl)
            # cv2.circle(mask, (c,r), radius=2, color=(0, 0, 255), thickness=4)
        # cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mask)
    print(rna_pth, len(df), f'no nucl {acc / len(df)}', 'done')


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
                        default=512,
                        help='Size of cropped single-cell image.')
    args = parser.parse_args()

    csv_file = ('tx', 'exprMat', 'metadata')
    img_type = (('CellComposite', 'jpg'),
                ('CellLabels', 'tif'),
                ('CellOverlay', 'jpg'),
                ('CompartmentLabels', 'tif'),
                # GeneLabels need to be processed
                ('GeneLabels', 'npz'))
    cell_comp = {'0': 0, 'Nuclear': 1,
                 'Membrane': 2, 'Cytoplasm': 3}
    # slide = {'1': 304,} 
    slide = {'2': 383}

    # this implementation is version sensitive
    # tiledb==0.20.0
    # tiledbsoma==0.1.22
    config = tiledb.Config()
    ctx = tiledb.Ctx(config)
    pySoma = tiledbsoma.SOMACollection(str(args.path / 'raw' / 'LiverDataRelease'),
                                       ctx=ctx)

    # Prepare single-cell images
    with multiprocessing.Pool(processes=args.core) as pool:
        test_args = list()
        df_cell = pySoma['RNA'].obs.df(attrs=['y_FOV_px', 'x_FOV_px'])
        df_cell = df_cell.rename_axis(None, axis=0)
        for s, fov in slide.items():
            pth = args.path / f'Liver{s}'
            crop_pth = pth / 'Debug'
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
                    test_args.append((df, f,
                                      args.crop_sz, args.fov_sz,
                                      crop_pth, pth))
        pool.starmap(test_cent, test_args)
