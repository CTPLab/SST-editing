import unittest
import scanpy as sc
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path


class TestXeniumMeta(unittest.TestCase):
    # pd.read_table is way faster
    def test_stat(self,
                  root=Path('Data/Xenium/')):
        cmp_dct = {'Gene Expression': 'transcript_counts',
                   'Negative Control Codeword': 'control_codeword_counts',
                   'Negative Control Probe': 'control_probe_counts',
                   # this is not very useful, only meant for
                   # simply assigning columns name for pd.read_table
                   'total_counts': 'total_counts'}

        for i in ('Rep1', 'Rep2'):
            meta_pth = root / i / 'meta'
            adata = sc.read_10x_h5(filename=str(meta_pth / 'cell_feature_matrix.h5'),
                                   gex_only=False)

            df_cell = pq.read_table(str(meta_pth / 'cells.parquet'),
                                    columns=['cell_id'] + list(cmp_dct.values())).to_pandas()
            # copy the cells.parquet to the scanpy data
            adata.obs = df_cell.copy()

            gene_dct = dict(adata.var['feature_types'])
            print(len(df_cell))
            df_cell.drop(df_cell[df_cell['total_counts'] == 0].index,
                         inplace=True)
            print(len(df_cell))

            df_gene = pq.read_table(str(meta_pth / 'transcripts.parquet'),
                                    filters=[('cell_id', '>=', 0),  # cell_id start with 1
                                             # https://www.10xgenomics.com/cn/resources/analysis-guides/performing-3d-nucleus-segmentation-with-cellpose-and-generating-a-feature-cell-matrix
                                             # critical QV thres >= 20
                                             ('qv', '>=', 20)],
                                    columns=['cell_id', 'feature_name']).to_pandas()

            # the following steps can be very slow
            print('start replacing the gene_dct')
            # convert binary gene name to string then to feature_types
            # for the following count comparison (~3000s)
            df_gene.feature_name = df_gene.feature_name.\
                str.decode('utf-8').map(gene_dct)

            print('start counting the exprs')
            # compute the counts for Gene expr, neg ctr prob,
            # neg ctr cod and blank code (can be slow)
            df_gene = df_gene.groupby(['cell_id', 'feature_name'],
                                      as_index=False).size()

            print('start transposing the df')
            # transpose the df format so it is similar to df_cell
            df_gene = df_gene.pivot(index='cell_id', columns='feature_name',
                                    values='size').reset_index()
            df_gene = df_gene.fillna(0).astype(int)
            df_gene.columns.name = None
            # map the expr names to count names by cmp_dct
            df_gene.rename(columns=cmp_dct, inplace=True)

            print('start summing the total counts')
            # get the total counts
            df_gene['total_counts'] = df_gene.iloc[:, 1:].sum(axis=1)
            df_gene = df_gene.drop(columns=['Blank Codeword'])
            df_diff = pd.concat([df_cell, df_gene]).\
                drop_duplicates(keep=False)
            print(len(df_cell), len(df_gene), len(df_diff))
            print(df_diff)


if __name__ == "__main__":
    unittest.main()
