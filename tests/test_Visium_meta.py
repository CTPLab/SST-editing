import unittest
import pandas as pd

from pathlib import Path


class TestVisiumMeta(unittest.TestCase):
    def test_stat(self,
                  path=Path('Data/Visium/10x_tap/')):
        df = pd.read_csv(str(path / '10x_image-paths_resolution.csv'))
        dfm = pd.read_csv(str(path.parent / 'GAN' / 'metadata.csv'))
        # the loop is slow, ~60s
        for rid, row in dfm.iterrows():
            if rid < 10:
                print(row.values)
            rep = row['replicate']
            assert rep in row['path']
            assert row['immune_phenotype'] == df.loc[df.replicate ==
                                                     rep, 'immune_phenotype'].values
            assert row['slide_name'] == df.loc[df.replicate ==
                                               rep, 'image_name'].values
            assert row['resolution'] == df.loc[df.replicate ==
                                               rep, 'resolution'].values
            assert row['patient'] == df.loc[df.replicate ==
                                            rep, 'sample'].values


if __name__ == "__main__":
    unittest.main()
