import pyvips
import sparse
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import scanpy as sc

from PIL import Image
from pathlib import Path


def prep_spot(adata_pth, slide_pth, out_pth,
              marker, crop, 
              resize=None,
              debug=False):
    # prepare the intermediate training data
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
        if debug:
            Image.fromarray(spot_np).save(str(out_pth / f'{nam}.jpg'))
        np.savez_compressed(str(out_pth / nam), img=spot_np,
                            key_melanoma_marker=np.array(cnt))
    print(slide_pth, 'done!')

def prep_crop(pth):
    # preprare the final training data that share the same format as others
    # 'Data/Visium/GAN/crop_spot/MELIPIT/MELIPIT-rep2/11x111.npz'
    npz = np.load(pth)
    img = npz['img'][96:-96, 96:-96]
    img = Image.fromarray(img)
    # the resize step is inspired by clean-FID
    img = img.resize((128, 128), resample=Image.Resampling.BICUBIC)
    img = np.asarray(img).clip(0, 255).astype(np.uint8)
    img_pth = pth.replace('.npz', '_hne.png').replace('crop_spot', 'crop')
    Image.fromarray(img).save(img_pth)

    npz_nm, npz_dir = Path(pth).name, Path(pth).parent
    gpth = npz_dir / f'{npz_dir.name}_{npz_nm}'
    gpth = str(gpth).replace('crop_spot', 'crop_old')
    gene_expr = np.load(gpth)
    gene_expr = sparse.COO.from_numpy(np.array(gene_expr['key_melanoma_marker']))
    gene_pth = pth.replace('.npz', '_rna.npz').replace('crop_spot', 'crop')
    sparse.save_npz(gene_pth, 
                    gene_expr)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path('Data/Visium/10x_tap'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--crop_size',
                        type=int,
                        default=128,
                        help='half size of cropped image.')
    parser.add_argument('--prep_spot',
                        action='store_true',
                        help='Prepare spot images and gene')
    parser.add_argument('--prep_crop',
                        action='store_true',
                        help='Prepare crop images for training')
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to visualize additional cell mask.')
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
    out_spot = (args.root.parent / 'GAN' / 'crop_spot')
    if args.prep_spot:
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = list()
            for index, row in df.iterrows():
                adata_pth = gene_pth / row['sample'] / row['replicate'] / \
                    'preprocessed.h5ad'  # 'filtered.h5ad'
                slide_pth = img_pth / row['image_name']
                out_pth = out_spot / row['sample'] / row['replicate']
                out_pth.mkdir(parents=True, exist_ok=True)
                if row['resolution'] == 0.1736:
                    # 0.3448 / 0.1736 ~ 2
                    resize, crop = 2, args.crop_size * 2
                else:
                    resize, crop = None, args.crop_size
                prep_args.append((adata_pth, slide_pth, out_pth,
                                  marker, crop, resize, args.debug))
            pool.starmap(prep_spot, prep_args)

    out_crop = (args.root.parent / 'GAN' / 'crop')
    if args.prep_crop:
        for index, row in df.iterrows():
            out_pth = out_crop / row['sample'] / row['replicate']
            out_pth.mkdir(parents=True, exist_ok=True)
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = out_spot.rglob('*.npz')
            prep_args = [(str(p),) for p in prep_args]
            pool.starmap(prep_crop, prep_args)
