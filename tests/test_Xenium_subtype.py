import cv2
import sparse
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
from pathlib import Path


def prep_mark_img(dct, clt, pth, hei=15, wid=30, shf=5):
    if (pth / f'marker_{clt}.png').is_file():
        return
    mark = np.ones([(hei+shf) * (len(dct) - 1), wid * 6, 3]) * 255
    for key, val in dct.items():
        if key != 0:
            k = key - 1
            mark[(hei+shf)*k:(hei+shf)*k + hei, :wid] = val
            cv2.putText(mark, str(key), (wid + 2, (hei+shf)*k + hei),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.imwrite(str(pth / f'marker_{clt}.png'), mark.astype(np.uint8))


def visual_subtype(df, msk_pth, out_pth):
    msk = sparse.load_npz(str(msk_pth))
    msk = msk[:, :, 0].todense()

    out = np.stack([(df.r.values)[msk],
                    (df.g.values)[msk],
                    (df.b.values)[msk]], axis=-1)

    pth = out_pth / msk_pth.name.replace('_msk.npz', '.png')
    cv2.imwrite(str(pth), out.astype(np.uint8))
    print(pth, 'done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test CosMx subtype.py')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/Xenium'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--cluster',
                        type=str,
                        default='graphclust',
                        help='folder to clustering tables: graphclust, kmeans_{2-10}_clusters')
    args = parser.parse_args()
    with multiprocessing.Pool(processes=args.core) as pool:
        # prepare color palatte
        if args.cluster == 'graphclust':
            # 29 for rep1, 21 for rep2
            clt_max = 29
        elif 'kmeans' in args.cluster:
            # Rep1, 2 have the same number of clusters.
            clt_max = int(args.cluster.split('_')[1])
        clr = sns.color_palette(cc.glasbey, n_colors=clt_max)
        clr_dct = {i + 1: (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                   for i, c in enumerate(clr)}
        clr_dct[0] = (0, 0, 0)
        prep_mark_img(clr_dct, args.cluster, args.path)

        sub_args, col_lst = list(), ['r', 'g', 'b']
        for r in ('Rep1', 'Rep2'):
            clt_pth = args.path / r / 'outs' / 'analysis' / 'clustering' / \
                f'gene_expression_{args.cluster}'
            df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
            bar_max = df_raw.Barcode.max()
            assert df_raw.Barcode.is_monotonic_increasing
            df = df_raw.set_index('Barcode').reindex(range(1, bar_max + 1),
                                                     fill_value=0)
            df = df.reset_index().astype(int)
            # df = df[df.Cluster != 0].reset_index(drop=True)
            # assert df_raw.equals(df)

            # split cluster color to three columns
            df.Cluster = df.Cluster.map(clr_dct)
            for n, col in enumerate(col_lst):
                df[col] = df.Cluster.apply(lambda x: x[n])
            df = df.drop(columns=['Cluster'])
            # insert (top) 0 list for assigning black to background
            df.loc[-1] = [0, 0, 0, 0]
            df.index = df.index + 1
            df = df.sort_index()
            assert df.Barcode.is_monotonic_increasing
            print(df, '\n')

            out_pth = args.path / r / f'type_{args.cluster}'
            out_pth.mkdir(parents=True, exist_ok=True)
            msk_pth = (args.path / r / 'rna').glob('*_msk.npz')
            for msk in msk_pth:
                sub_args.append((df, msk, out_pth))
                # break
        pool.starmap(visual_subtype, sub_args)
