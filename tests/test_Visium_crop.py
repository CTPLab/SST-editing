import sparse
import unittest
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path


class TestVisiumCrop(unittest.TestCase):
    def test_stat(self,
                  path=Path('Data/Visium/')):
        df = pd.read_csv(
            str(path / '10x_tap' / '10x_image-paths_resolution.csv'))
        out_pth = path / 'GAN' / 'crop'

        for index, row in df.iterrows():
            print(row.values)
            spot_npz = (out_pth / row['sample'] /
                        row['replicate']).rglob('*.npz')
            for spot in spot_npz:
                spot_dt0 = sparse.load_npz(str(spot)).todense()
                spot_name = str(spot.parent /
                                f'{spot.parent.name}_{spot.name}')
                spot_dt1 = np.load(spot_name.replace('crop', 'crop_old').replace('_rna.npz', '.npz'))
                assert (spot_dt0 ==
                        np.array(spot_dt1['key_melanoma_marker'])).all()
                if row['resolution'] == 0.3448:
                    img_pth = str(spot).replace('rna.npz', 'hne.png')
                    img = np.array(Image.open(img_pth))
                    img1 = spot_dt1['img'][96:-96, 96:-96]
                    img1 = Image.fromarray(img1)
                    # the resize step is inspired by clean-FID
                    img1 = img1.resize((128, 128), resample=Image.Resampling.BICUBIC)
                    img1 = np.asarray(img1).clip(0, 255).astype(np.uint8)
                    if not (img == img1).all():
                        print('img not consistent', spot)


if __name__ == "__main__":
    unittest.main()
