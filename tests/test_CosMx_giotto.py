import unittest
import numpy as np
import pandas as pd
from pathlib import Path


class TestCosMxGiotto(unittest.TestCase):
    def test_stat(self,
                  path=Path('Data/CosMx/')):
        col = ['Area', 'AspectRatio', 'Width',
               'Height', 'Mean.DAPI', 'Max.DAPI']
        df_rna = pd.read_csv(path / 'Giotto' / 'rna.csv')
        tissue = np.unique(df_rna.Run_Tissue_name)
        print(df_rna.columns)
        print(np.unique(df_rna.cell_type))
        print(np.unique(df_rna.niche))
        for t in tissue:
            dfr = df_rna[df_rna.Run_Tissue_name == t]
            meta_pth = path / t / f'{t}-Flat_files_and_images'
            dfm = pd.read_csv(str(meta_pth / f'{t}_metadata_file.csv'))
            tot0, tot1 = len(dfr), 0
            print(t, tot0, len(dfm))
            for f in np.unique(dfr.fov):
                dr = dfr[dfr.fov == f]
                dr_cid = np.unique(dr.cell_ID)
                dr_cid = [int(cid.split('_')[-1]) for cid in dr_cid]
                dm = dfm[(dfm.fov == f) & (dfm.cell_ID.isin(dr_cid))]

                dr = dr[col].reset_index(drop=True)
                dm = dm[col].reset_index(drop=True)
                if not dr.equals(dm):
                    print(t, f)
                tot1 += len(dm)
            print(f'{tot0}, {tot1} \n')


if __name__ == "__main__":
    unittest.main()
