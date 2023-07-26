import sys
import cv2
import pickle
import random
import sparse
import tiledb
import argparse
import itertools
import tiledbsoma

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path
from random import shuffle
from tifffile import imread
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


def test_imread(arr, chn, 
                pth, save_pth, 
                sz, fov_sz):
    flo_img = imread(str(pth))[chn]
    flo_img = flo_img.transpose((1, 2, 0))
    # flo_img = flo_img.astype(np.float32) / flo_img.max(axis=(0, 1))
    # flo_img = (flo_img * 255).astype(np.uint8)
    msk_fov = pth.stem.split('_')[-1]
    # msk_pth = pth.parent.parent / 'CellOverlay' / f'CellOverlay_{msk_fov}.jpg'
    # msk_img = cv2.imread(str(msk_pth), flags=cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(str(save_pth / f'flo_{msk_fov}.jpg'), flo_img[512:, 512:])
    # cv2.imwrite(str(save_pth / f'msk_{msk_fov}.jpg'), msk_img[512:, 512:])
    out, sz = [], sz // 2
    for rid, row in enumerate(arr):
        r, c = int(row[0]), int(row[1])
        if sz + 512 <= r <= fov_sz - sz and \
           sz + 512 <= c <= fov_sz - sz:
            flo = flo_img[r - sz: r + sz, c - sz: c + sz]
            flo = np.concatenate((np.zeros_like(flo[:, :, 0])[:, :, None], flo), -1)
            flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
            # msk = msk_img[r - sz: r + sz, c - sz: c + sz]
            cv2.imwrite(str(save_pth / f'{r}_{c}_{msk_fov}_{rid}_flo.png'), flo)
            # cv2.imwrite(str(save_pth / f'{r}_{c}_{msk_fov}_{rid}_msk.jpg'), msk)
    print(pth, 'done')


def gen_stat1(pth):
    im = cv2.imread(str(pth), flags=cv2.IMREAD_UNCHANGED)
    out = np.median(im, axis=(0, 1)).tolist() + np.mean(im, axis=(0, 1)).tolist()
    out += np.min(im, axis=(0, 1)).tolist() + np.max(im, axis=(0, 1)).tolist()
    return out


def gen_stat(arr, pth,
             sz, fov_sz):
    # cd298/b2m channel
    flo_img = imread(str(pth))

    out, sz = [], sz // 2
    for _, row in enumerate(arr):
        r, c = int(row[0]), int(row[1])
        if sz + 512 <= r <= fov_sz - sz and \
           sz + 512 <= c <= fov_sz - sz:
            med = np.median(flo_img[:, r - sz: r + sz, c - sz: c + sz], axis=(1, 2))
            avg = np.mean(flo_img[:, r - sz: r + sz, c - sz: c + sz], axis=(1, 2))
            mn = np.min(flo_img[:, r - sz: r + sz, c - sz: c + sz], axis=(1, 2))
            mx = np.max(flo_img[:, r - sz: r + sz, c - sz: c + sz], axis=(1, 2))
            out.append((row[2:].tolist() + med.tolist() + \
                        avg.tolist() + mn.tolist() + mx.tolist()))
    print(pth, flo_img.shape, arr.shape, 'done')
    return out


def gen_crop(df, gene, fov,
             sz, fov_sz,
             pth, fov_pth,
             debug=True,
             prefx='GeneLabels',
             num=100000):
    # rna expr image show boundary effects
    # e.g., minor missing gene expr for row=64, ..., 70
    # Compartment label can be tricky,
    # e.g., Membrane represent non-smooth, twisted small regions,
    # which may not be useful.
    rna_pth = fov_pth / f'{prefx}' / f'{prefx}_F{str(fov).zfill(3)}'
    rna_coo = sparse.load_npz(str(rna_pth) + '.npz')
    flo_pth = str(rna_pth).replace(prefx, 'CellComposite')
    flo_img = cv2.imread(flo_pth + '.jpg', flags=cv2.IMREAD_UNCHANGED)
    # comp_pth = str(rna_pth).replace(prefx, 'CompartmentLabels')
    # comp_img = cv2.imread(comp_pth + '.tif', flags=cv2.IMREAD_UNCHANGED)
    # cell_pth = str(rna_pth).replace(prefx, 'CellLabels')
    # cell_img = cv2.imread(cell_pth + '.tif', flags=cv2.IMREAD_UNCHANGED)

    df_sub = zip(df.index, df.y_FOV_px, df.x_FOV_px, df[gene])
    if debug:
        df_sub = list(df_sub)
        shuffle(df_sub)
        # prep the raw cell mask which has minor shift
        mask_pth = str(rna_pth).replace(prefx, 'CellOverlay')
        mask = cv2.imread(mask_pth + '.jpg', flags=cv2.IMREAD_UNCHANGED)
        # improve the visualization
        rna_coo *= 20000
        # comp_img *= 50
        # cell_img *= 50
        # cell_img += np.sign(cell_img) * 20000

    acc, sz = 0, sz // 2
    for _, (cid, r, c, g) in enumerate(df_sub):
        if sz + 512 <= r <= fov_sz - sz and \
           sz + 512 <= c <= fov_sz - sz:
            rna = rna_coo[r - sz: r + sz, c - sz: c + sz]
            flo = flo_img[r - sz: r + sz, c - sz: c + sz]
            # comp_lab = comp_img[r - sz: r + sz, c - sz: c + sz]
            # cell_lab = cell_img[r - sz: r + sz, c - sz: c + sz]
            if debug:
                # cv2.imwrite(str(pth / f'{cid}_tot.png'),
                #             rna.sum(axis=-1).todense().astype(np.uint16))
                cv2.imwrite(str(pth / f'{cid}_{g}flo.jpg'), flo)
                mk = mask[r - sz: r + sz, c - sz: c + sz].copy()
                cv2.circle(mk, (sz, sz), radius=2,
                           color=(0, 0, 255), thickness=2)
                cv2.imwrite(str(pth / f'{cid}_{g}msk.jpg'), mk)

                # cp = comp_lab.copy()
                # cl = cell_lab.copy().astype(np.uint16)
                # cv2.imwrite(str(pth / f'{cid}_comp.png'), cp)
                # cv2.imwrite(str(pth / f'{cid}_cell.png'), cl)

                # cv2.circle(mask, (c,r), radius=2, color=(0, 0, 255), thickness=4)
                acc += 1
                if acc >= num:
                    # cv2.imwrite(str(pth / f'{cid}_msk.jpg'), mask)
                    break
            else:
                sparse.save_npz(str(pth / f'{cid}_rna'), rna)
                cv2.imwrite(str(pth / f'{cid}_flo.png'), flo)
                # cv2.imwrite(str(pth / f'{cid}_comp.png'), comp_lab)
                # cv2.imwrite(str(pth / f'{cid}_cell.png'),
                            # cell_lab.astype(np.uint16))
    print(rna_pth, len(df), 'done')
    return 


def gen_img(img_pth, save_pth):
    print(img_pth)
    img = cv2.imread(str(img_pth), flags=cv2.IMREAD_UNCHANGED)
    img_nm = img_pth.stem
    cv2.imwrite(str(save_pth / f'{img_nm}.jpg'), img[512:, 512:])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/CosMx'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--fov_sz',
                        type=int,
                        default=4256,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--crop_sz',
                        type=int,
                        default=160,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--gene',
                        type=str,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--topn',
                        type=int,
                        default=200,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--test_imread',
                        action='store_true',
                        help='Test the imread function.')
    parser.add_argument('--test_stat',
                        action='store_true',
                        help='test the min, max, median stat calc.')
    parser.add_argument('--output_stat',
                        action='store_true',
                        help='create the plots for visualization.')
    parser.add_argument('--prep_img',
                        action='store_true',
                        help='Prepare the cropped pair of cell and gene images.')
    parser.add_argument('--prep_crop',
                        action='store_true',
                        help='Prepare the cropped pair of cell and gene images.')
    parser.add_argument('--prep_stat',
                        action='store_true',
                        help='Prepare the cropped pair of cell and gene images.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to visualize additional cell mask.')
    args = parser.parse_args()


    if args.test_imread:
        slide = {1:304, 2: 383}
        with multiprocessing.Pool(processes=args.core) as pool:
            gen_args = list()
            df_cell = pd.read_csv('Data/CosMx/GAN/crop/metadata.csv')
            for s, fov in slide.items():
                pth = args.path / f'Liver{s}' 
                save_pth = pth / 'debug_imread'
                save_pth.mkdir(parents=True, exist_ok=True)
                pth = pth / 'Morphology2D_Normalized' / f'20221008_005902_S{s + 1}_C902_P99_N99'
                cnd = df_cell['slide_ID_numeric'] == s
                dfc = df_cell[cnd]
                for f in range(1, fov + 1):
                    df = dfc[dfc.fov == f]
                    pthf = pth.parent / f'{pth.name}_F{str(f).zfill(3)}.TIF'
                    if df.empty:
                        print(f, 'empty')
                    else:
                        gen_args.append((df[['y_FOV_px', 'x_FOV_px']].to_numpy(), [2, 4],
                                         pthf, save_pth,
                                         args.crop_sz, args.fov_sz))
            pool.starmap(test_imread, gen_args)

    if args.test_stat:
        # Prepare single-cell images
        with multiprocessing.Pool(processes=args.core) as pool:
            pth = (args.path / 'Liver1'/'debug_imread').glob('*.png') 
            test_args = [(p,) for p in list(pth)]
            out = pool.starmap(gen_stat1, test_args)
            out = np.array(out)
            print(out.shape)
            np.save('stats/0_cosmx_1.npy', out)

    csv_file = ('tx', 'exprMat', 'metadata')
    img_type = (('CellComposite', 'jpg'),
                ('CellLabels', 'tif'),
                ('CellOverlay', 'jpg'),
                ('CompartmentLabels', 'tif'),
                # GeneLabels need to be processed
                ('GeneLabels', 'npz'))
    cell_comp = {'0': 0, 'Nuclear': 1,
                 'Membrane': 2, 'Cytoplasm': 3}
    slide = {'1': 304, '2': 383}

    if args.prep_img:
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = list()
            for i in ('Liver1', 'Liver2'):
                data_pth = args.path / i
                save_pth = data_pth / 'CellComposite_t'
                save_pth.mkdir(parents=True, exist_ok=True)
                img_pth = list((data_pth / 'CellComposite').glob('*.jpg'))
                img_pth.sort()
                for p in img_pth[:100]:
                    prep_args.append((p, save_pth))
            pool.starmap(gen_img, prep_args)


    stype = 'tumor'
    if args.prep_crop:
        # Prepare single-cell images
        with multiprocessing.Pool(processes=args.core) as pool:
            gen_args = list()
            df_cell = pd.read_csv('Data/CosMx/GAN/crop/metadata.csv')
            df_gene = pd.read_csv('Data/CosMx/GAN/crop/metadata_img.csv')
            # df_cell = pySoma['RNA'].obs.df(attrs=['y_FOV_px', 'x_FOV_px'])
            # df_cell = df_cell.rename_axis(None, axis=0)
            for s, fov in slide.items():
                pth = args.path / f'Liver{s}'
                if args.debug:
                    crop_pth = pth / f'debug_pixel_{args.gene}_{stype}_{args.topn}'
                else:
                    crop_pth = pth.parent / 'GAN' / 'crop' / pth.name
                crop_pth.mkdir(parents=True, exist_ok=True)
                print(crop_pth)

                subtype = 'Hep' if s == '1' else stype
                cnd = (df_cell['slide_ID_numeric'] == int(s)) & (df_cell['cellType'].str.contains(subtype))
                dfc = df_cell[cnd]
                dfg = df_gene[cnd]
                if args.topn > 0:
                    nlg = dfg.nlargest(args.topn, args.gene)
                elif args.topn < 0:
                    nlg = dfg.nsmallest(-args.topn, args.gene)
                dfc = dfc[dfc.index.isin(nlg.index)]
                dfg = dfg[dfg.index.isin(nlg.index)]
                dfc[args.gene] = dfg[args.gene]
                for f in range(1, fov + 1):
                    df = dfc[dfc.fov == f]
                    if df.empty:
                        npz_pth = pth / 'GeneLabels' / \
                            f'GeneLabels_F{str(f).zfill(3)}.npz'
                        print(f, npz_pth.is_file(), 'empty')
                    else:
                        gen_args.append((df, args.gene, f,
                                         args.crop_sz, args.fov_sz,
                                         crop_pth, pth,
                                         args.debug))
            pool.starmap(gen_crop, gen_args)

    slide = {2: 383}
    # slide = {1: 304}
    gene = ['HLA-A', 'APOA1', 'TTR', 'MALAT1', 'B2M', 'APOC1', 'APOE', 
            'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRA', 'HLA-DRB1']
    stype = 'Hep'
    suffix = '' if stype is None else f'_{stype}'
    if args.prep_stat:
        # Prepare single-cell images
        with multiprocessing.Pool(processes=args.core) as pool:
            gen_args = list()
            df_cell = pd.read_csv('Data/CosMx/GAN/crop/metadata.csv')
            df_gene = pd.read_csv('Data/CosMx/GAN/crop/metadata_img.csv')
            for s, fov in slide.items():
                pth = args.path / f'Liver{s}' 
                pth = pth / 'Morphology2D_Normalized' / f'20221008_005902_S{s + 1}_C902_P99_N99'
                cnd = df_cell['slide_ID_numeric'] == s
                if stype is not None:
                    cnd = cnd & (df_cell['cellType'].str.contains(stype))
                dfc = df_cell[cnd]
                dfg = df_gene[cnd]
                print(len(dfc))
                for gn in gene:
                    dfc[gn] = dfg[gn]
                dfc = dfc[['fov', 'y_FOV_px', 'x_FOV_px'] + gene]
                for f in range(1, fov + 1):
                    df = dfc[dfc.fov == f]
                    pthf = pth.parent / f'{pth.name}_F{str(f).zfill(3)}.TIF'
                    if df.empty:
                        print(f, 'empty')
                    else:
                        gen_args.append((df[['y_FOV_px', 'x_FOV_px'] + gene].to_numpy(), 
                                         pthf,
                                         args.crop_sz, args.fov_sz))
            result = pool.starmap(gen_stat, gen_args)
            result = np.array(sum(result, []))
            print(result.shape)
            col_lst = gene + [f'median{i}' for i in range(5)] + [f'mean{i}' for i in range(5)] + \
                             [f'min{i}' for i in range(5)] + [f'max{i}' for i in range(5)]
            df = pd.DataFrame({col:result[:,i] for i, col in enumerate(col_lst)})
            df.to_csv(f'stats/0_cosmx_{list(slide.keys())[0]}{suffix}.csv')

    # Hep df['min2'].quantile(0.05), df['median2'].median(), df['max2'].quantile(0.95)
    # 26.0 91.0 595.0
    # 40.0 183.0 716.05
    # None df['min2'].quantile(0.05), df['median2'].median(), df['max2'].quantile(0.95)
    # 23.0 91.0 631.0
    # 90.0 379.0 1550.0
    # normal HLA in normal:  
    # 8.91563343201407 5.539585919900612
    # normal HLA in tumor:  
    # 29.544871794871796 15.278897555447463

    # CD298/B2M
    # 23.0 91.0 631.0
    # 90.0 379.0 1550.0 
    # Dapi
    # 12.0 92.0 1842.0
    # 24.0 387.0 2375.0
    if args.output_stat:
        stype = 'Hep'
        suffix = '' if stype is None else f'_{stype}'
        for i in (1, 2):
            title = 'normal' if i == 1 else 'tumor'
            df = pd.read_csv(f'stats/0_cosmx_{i}{suffix}.csv')
            print(len(df))
            print(df['min2'].min(), df['min2'].quantile(0.05), df['median2'].median(), df['max2'].quantile(0.95), df['max2'].max())
            print(df['min4'].min(), df['min4'].quantile(0.05), df['median4'].median(), df['max4'].quantile(0.95), df['max4'].max())
            # print(df['HLA-A'].mean(), df['HLA-A'].std())
            for (x, y) in list(itertools.product(['HLA-DRB1',], ['median2', 'median4'])):
                print(x, y)
                sns.lmplot(data=df, x=x, y=y, 
                        scatter_kws={'s': 0.5, 'alpha':0.1})
                plt.title(title)
                plt.savefig(f'plot_cosmx/cosmx{suffix}_{title}_{x}_{y}.png')
                plt.close()
        