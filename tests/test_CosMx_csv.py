import cv2
import sparse
import argparse

import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test CosMx csv file')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/CosMx/'),
                        help='Path to NanoString dataset.')
    args = parser.parse_args()

    df_csv = pd.read_csv(args.path / 'GAN' / 'metadata1.csv')
    col = ['Area', 'AspectRatio', 'Width',
            'Height', 'Mean.DAPI', 'Max.DAPI']
    df_rna = pd.read_csv(args.path / 'Giotto' / 'rna.csv')
    tissue = np.unique(df_rna.Run_Tissue_name)
    for t in tissue:
        dfc = df_csv[df_csv.tissue == t]
        dfr = df_rna[df_rna.Run_Tissue_name == t]
        meta_pth = args.path / t / f'{t}-Flat_files_and_images'
        dfm = pd.read_csv(str(meta_pth / f'{t}_metadata_file.csv'))
        tot1 = 0
        print(t, len(dfc), len(dfr), len(dfm))
        for f in np.unique(dfr.fov):
            dc = dfc[dfc.fov == f]
            # only the cell_ID that includes in rna.csv 
            # is assigned with cell type and nich 
            dc = dc.dropna()
            dr = dfr[dfr.fov == f]
            dr_cid = np.unique(dr.cell_ID)
            dr_cid = [int(cid.split('_')[-1]) for cid in dr_cid]
            dm = dfm[(dfm.fov == f) & (dfm.cell_ID.isin(dr_cid))]
            dr = dr[col].reset_index(drop=True)
            dm = dm[col].reset_index(drop=True)
            assert dr.equals(dm) and dc.cell_ID.isin(dr_cid).all()
            tot1 += len(dm)
        print(f'{tot1} \n')
    