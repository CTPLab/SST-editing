import cv2
import sparse
import argparse

import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Xenium csv file')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/Xenium/'),
                        help='Path to NanoString dataset.')
    args = parser.parse_args()

    df_meta = pd.read_csv(args.path / 'GAN' / 'metadata.csv')
    for r in ('Rep1', 'Rep2'):
        df = df_meta[df_meta.rep == int(r[-1])]
        print(len(df))
        for clt in range(11):
            if clt == 0:
                clt_nam = 'graphclust'
            if clt == 1:
                continue
            if clt > 1:
                clt_nam = f'kmeans_{clt}_clusters'
            print(clt_nam)

            clt_pth = args.path / r / 'outs' / 'analysis' / 'clustering' / \
                f'gene_expression_{clt_nam}'
            df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
            df_dct = dict(zip(df_raw.Barcode, df_raw.Cluster))
            df_itr = zip(df.cell_id, df[clt_nam])
            for cid, clt in df_itr:
                if cid not in df_dct:
                    if clt != 0:
                        print('w/o clt', clt_nam, cid, clt)
                else:
                    if clt != df_dct[cid]:
                        print('w/t clt', clt_nam, cid, clt)
