import zarr
import unittest
import numpy as np
import scanpy as sc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from Dataset.utils import _um_to_pixel, _df_to_roi

# suppress chained assignment warning
pd.options.mode.chained_assignment = None

# Conclusion: ~1/10000 rnas have inconsistent assigned cell_id
# in transcripts.parquet and cells.zarr.zip


class TestXeniumRoi(unittest.TestCase):
    def test_stat(self,
                  root=Path('Data/Xenium/')):
        rna_col = ['cell_id', 'y_location', 'x_location',
                   'overlaps_nucleus']
        msk_nam = ('nucl', 'cell')

        for i in ('Rep1', 'Rep2'):
            meta_pth = root / i / 'meta'
            with zarr.ZipStore(str(meta_pth / 'cells.zarr.zip'), mode='r') as store:
                meta_msk = zarr.group(store=store, overwrite=False).masks
                df_rna = pq.read_table(str(meta_pth / 'transcripts.parquet'),
                                       filters=[('cell_id', '>=', 0),
                                                ('qv', '>=', 20)],
                                       columns=rna_col).to_pandas()
                # convert um (float) to pixel (int)
                _um_to_pixel(df_rna, '{}_location')

                roi_pth = root / i / 'hne'
                roi_lst = list(roi_pth.glob('*.jpg'))

                for rid, roi in enumerate(roi_lst):
                    # no duplicated cell_id or no overlap centroid for different id
                    crd_all = list(map(int, roi.stem.split('_')[:-1]))
                    crd, crdo = crd_all[:4], crd_all[4:]
                    df_rn = _df_to_roi(df_rna, crd, crd, '{}_location')
                    msk_np = {'nucl': meta_msk[0][crd[0]:crd[1], crd[2]:crd[3]],
                              'cell': meta_msk[1][crd[0]:crd[1], crd[2]:crd[3]]}
                    # msk_np = np.load(str(roi))
                    for mid, msk in enumerate(msk_nam):
                        df_r = df_rn[df_rn.overlaps_nucleus == 1 - mid].\
                            drop(columns=['overlaps_nucleus'])

                        cid0 = df_r.cell_id.values
                        cid1 = msk_np[msk][df_r.y_location.values,
                                           df_r.x_location.values]
                        cnd = (cid0 != cid1)
                        print(msk, cnd.sum(), len(cid0))
                    print('\n')


if __name__ == "__main__":
    unittest.main()
