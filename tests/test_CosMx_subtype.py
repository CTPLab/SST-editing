import cv2
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
from pathlib import Path


def prep_mark_img(dct_t, dct_n, pth, hei=15, wid=30, shf=5):
    if (pth / 'marker_type.png').is_file() and (pth / 'marker_nich.png').is_file():
        return
    print(dct_t)
    print(dct_n)
    mark_t = np.ones([(hei+shf) * len(dct_t), wid * 6, 3]) * 255
    mark_n = np.ones([(hei+shf) * len(dct_n), wid * 9, 3]) * 255
    for kid, key in enumerate(dct_t.keys()):
        mark_t[(hei+shf)*kid:(hei+shf)*kid + hei, :wid] = dct_t[key]
        cv2.putText(mark_t, key, (wid + 2, (hei+shf)*kid + hei),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    for kid, key in enumerate(dct_n.keys()):
        mark_n[(hei+shf)*kid:(hei+shf)*kid + hei, :wid] = dct_n[key]
        cv2.putText(mark_n, key, (wid + 2, (hei+shf)*kid + hei),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.imwrite(str(pth / 'marker_type.png'), mark_t.astype(np.uint8))
    cv2.imwrite(str(pth / 'marker_nich.png'), mark_n.astype(np.uint8))


def visual_subtype(df, comp_pth,
                   cid_pth, out_pth):
    comp = cv2.imread(str(comp_pth), flags=cv2.IMREAD_UNCHANGED)
    cid = cv2.imread(str(cid_pth), flags=cv2.IMREAD_UNCHANGED)
    # only color cell type in nuclear region
    cid[comp != 1] = 0

    lst_t = [[], [], []]
    lst_n = [[], [], []]
    clr_t = dict(zip(df.cell_ID, df.cell_type))
    clr_n = dict(zip(df.cell_ID, df.niche))
    for c in range(cid[:].max() + 1):
        for i in range(3):
            if c not in clr_t:
                lst_t[i].append(0)
                lst_n[i].append(0)
            else:
                lst_t[i].append(clr_t[c][i])
                lst_n[i].append(clr_n[c][i])
    lst_t = np.array(lst_t)
    lst_n = np.array(lst_n)

    out_t = np.stack([lst_t[0][cid], lst_t[1][cid], lst_t[2][cid]], axis=-1)
    out_n = np.stack([lst_n[0][cid], lst_n[1][cid], lst_n[2][cid]], axis=-1)

    pth_t = out_pth.parent / f'type_{out_pth.name}'
    pth_n = out_pth.parent / f'nich_{out_pth.name}'
    cv2.imwrite(str(pth_t), out_t.astype(np.uint8))
    cv2.imwrite(str(pth_n), out_n.astype(np.uint8))
    print(cid_pth, 'done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Xenium subtype.py')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/Xenium'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    args = parser.parse_args()
    with multiprocessing.Pool(processes=args.core) as pool:
        col = ['cell_ID', 'cell_type', 'niche']
        sub_args = list()
        dfr = pd.read_csv(args.path / 'Giotto' / 'rna.csv')
        print(dfr[col].head(20))

        ctype = np.unique(dfr.cell_type)
        ctype_clr = sns.color_palette(cc.glasbey, n_colors=len(ctype))
        ctype_clr = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                     for c in ctype_clr]
        ctype_dct = {n: ctype_clr[i] for i, n in enumerate(ctype)}
        cnich = np.unique(dfr.niche)
        cnich_clr = sns.color_palette(cc.glasbey, n_colors=len(cnich))
        cnich_clr = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                     for c in cnich_clr]
        cnich_dct = {n: cnich_clr[i] for i, n in enumerate(cnich)}

        dfr.cell_ID = dfr.cell_ID.map(lambda x: int(x.split('_')[-1]))
        dfr.cell_type = dfr.cell_type.map(ctype_dct)
        dfr.niche = dfr.niche.map(cnich_dct)
        print(dfr[col].head(20))

        prep_mark_img(ctype_dct, cnich_dct, args.path)

        for t in np.unique(dfr.Run_Tissue_name):
            df = dfr[dfr.Run_Tissue_name == t]
            img_pth = args.path / t / f'{t}-Flat_files_and_images'
            (img_pth / 'Subtype').mkdir(parents=True, exist_ok=True)
            for f in np.unique(df.fov):
                cid_pth = img_pth / 'CellLabels' / \
                    f'CellLabels_F{str(f).zfill(3)}.tif'
                comp_pth = img_pth / 'CompartmentLabels' / \
                    f'CompartmentLabels_F{str(f).zfill(3)}.tif'
                out_pth = img_pth / 'Subtype' / f'F{str(f).zfill(3)}.png'
                sub_args.append((df[df.fov == f], comp_pth,
                                 cid_pth, out_pth))
        pool.starmap(visual_subtype, sub_args)
