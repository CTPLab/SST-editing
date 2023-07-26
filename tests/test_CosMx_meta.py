import unittest
import numpy as np
import pandas as pd
from pathlib import Path


class TestCosMxMeta(unittest.TestCase):
    def test_stat(self,
                  path=Path('Data/CosMx/')):
        # the whole test takes ~34805.910s
        for fld in path.iterdir():
            if fld.is_dir() and 'Lung' in str(fld):
                cell_fld = fld / f'{fld.name}-Flat_files_and_images'
                print(cell_fld)
                df_expr = pd.read_csv(cell_fld / f'{fld.name}_exprMat_file.csv',
                                      encoding='ISO-8859-1')

                # sort the rna expression name
                cell_id = df_expr.columns[0:2].tolist()
                expr_nm = np.sort(df_expr.columns.difference(cell_id)).tolist()
                # convert rna expr df to np array
                np_expr = df_expr.loc[:, cell_id+expr_nm].values
                print(np_expr.shape)

                df_tx = pd.read_csv(cell_fld / f'{fld.name}_tx_file.csv',
                                    encoding='ISO-8859-1', low_memory=False)
                print(df_tx.head)

                for r in range(len(np_expr)):
                    # for each cell, get the sorted rna expr from tx_file.csv
                    df = df_tx[(df_tx['fov'] == np_expr[r, 0]) &
                               (df_tx['cell_ID'] == np_expr[r, 1])].target.\
                        value_counts().reindex(expr_nm, fill_value=0).sort_index()
                    # Lung5_Rep2 fov=4, cell_ID=2968 seems incorrect
                    # df_tx[945] = 21, df_expr[945] = 22
                    # Lung5_Rep2_tx_file.csv requires encoding='ISO-8859-1', low_memory=False
                    if not np.all(df.values == np_expr[r, 2:]):
                        print(fld.name, np_expr[r, :2])
                    if (r + 1) % 5000 == 0:
                        print(r, np_expr[r, :2])


if __name__ == "__main__":
    unittest.main()
