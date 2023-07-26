import pyvips
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from PIL import Image
from pathlib import Path


def prep_roi(adata_pth, slide_pth, out_pth,
             marker, crop, resize=None):
    adata = sc.read_h5ad(str(adata_pth))
    sld = pyvips.Image.new_from_file(str(slide_pth),
                                     access='sequential')
    print(slide_pth, len(adata.layers['norm']),
          sld.height, sld.width, sld.bands, crop, resize)
    sld_np = np.ndarray(buffer=sld.write_to_memory(),
                        dtype=np.uint8,
                        shape=[sld.height, sld.width, sld.bands])

    gene_lst = list(adata.var['symbol'])
    midx = [gene_lst.index(m) for m in marker]
    spot = zip(adata.obs['spot'], adata.obsm['spatial'], adata.layers['norm'])
    for sid, (nam, crd, cnt) in enumerate(spot):
        cnt = cnt[midx]
        spot_np = sld_np[crd[0]-crop:crd[0]+crop,
                         crd[1]-crop:crd[1]+crop]
        if resize is not None:
            spt = Image.fromarray(spot_np)
            new_sz = crop * 2 // resize
            # the resize step is inspired by clean-FID
            spt = spt.resize((new_sz, new_sz),
                             resample=Image.Resampling.BICUBIC)
            spot_np = np.asarray(spt).clip(0, 255).astype(np.uint8)
        assert len(cnt) == 61
        Image.fromarray(spot_np).save(str(out_pth / f'{nam}.jpg'))
        # np.savez_compressed(str(out_pth / nam), img=spot_np,
        #                     key_melanoma_marker=np.array(cnt))
        if sid == 19:
            break
    print(slide_pth, 'done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path('Data/Visium/10x_tap'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--crop_size',
                        type=int,
                        default=128,
                        help='half size of cropped image.')
    args = parser.parse_args()

    marker = ['SLC2A1', 'CCN1', 'ATP1A1', 'S100A1', 'NES', 'SLC4A5', 'PAX3', 'MLPH', 'SEMA3B', 'WNT5A',
              'MITF', 'ROPN1B', 'SLIT2', 'SLC45A2', 'TGFBI', 'GFRA3', 'PDGFRB', 'ABCB5', 'AQP1', 'EGFR',
              'TMEM176B', 'GFRA2', 'LOXL2', 'MLANA', 'TYRP1', 'TNC', 'VIM', 'LOXL4', 'PLEKHB1', 'RAB38',
              'TYR', 'SLC2A3', 'PMEL', 'CDK2', 'ERBB3', 'NT5DC3', 'POSTN', 'SLC22A17', 'SERPINA3', 'AKT1',
              'CAPN3', 'CDH1', 'CDH13', 'NGFR', 'SOX9', 'CDH2', 'TCF4', 'BCL2', 'CDH19', 'MBP', 'MIA',
              'AXL', 'BIRC7', 'S100B', 'PRAME', 'SOX10', 'GPR143', 'GPM6B', 'PIR', 'GJB1', 'BGN']

    df = pd.read_csv(str(args.root / '10x_image-paths_resolution.csv'))

    img_pth = (args.root / 'tif_files')
    gene_pth = (args.root / 'data_derived')
    out_root = (args.root.parent / 'GAN' / 'crop_test')
    for index, row in df.iterrows():
        adata_pth = gene_pth / row['sample'] / row['replicate'] / \
            'preprocessed.h5ad'  # 'filtered.h5ad'
        slide_pth = img_pth / row['image_name']
        out_pth = out_root / row['sample'] / row['replicate']
        out_pth.mkdir(parents=True, exist_ok=True)
        if row['resolution'] == 0.1736:
            # 0.3448 / 0.1736 ~ 2
            resize, crop = 2, args.crop_size * 2
        else:
            resize, crop = None, args.crop_size

        prep_roi(adata_pth, slide_pth, out_pth,
                 marker, crop, resize)
        print('\n')
