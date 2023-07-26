
import sys
import cv2
import math
import zarr
import pickle
import random
import sparse
import argparse
import colorsys
import multiprocessing

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow.compute as pc
import pyarrow.parquet as pq

from math import ceil
from pathlib import Path
from random import shuffle
from PIL import Image, ImageFile
from tifffile import imread, imwrite
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from Dataset.utils import _um_to_pixel, _df_to_roi

ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


def prep_roi_img1(img_pth, out_pth, roi, ovlp=200, gamma=0.5, coef=1):
    img = imread(str(img_pth))

    is_rgb = len(img.shape) == 3
    hei, wid = img.shape[-2], img.shape[-1]
    h_num, w_num = ceil(hei / roi), ceil(wid / roi)
    print(h_num, w_num)
    for h in range(h_num):
        for w in range(w_num):
            print(h, w)
            crd, crdo = _roi_to_coord(h, w, hei, wid, roi, ovlp)
            print(crd, crdo, crd+crdo)
            if is_rgb:
                img_c = img[:, crdo[0]: crdo[1], crdo[2]: crdo[3]]
                img_c = img_c.transpose((1, 2, 0))
                img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)
            else:
                img_c = img[crdo[0]: crdo[1], crdo[2]: crdo[3]]

            roi_nm = '_'.join(map(str, crd + crdo))
            cv2.imwrite(str(out_pth / f'{roi_nm}.png'), 
                        img_c)


def prep_crop1(rna_pth, out_pth, df_mn, sz, debug=False, denum=10, gene_num=392):
    rna_coo = sparse.load_npz(str(rna_pth))
    dapi_pth = rna_pth.parent.parent / 'dapi_test' / (rna_pth.stem[:-4] + '.png')
    dapi_img = cv2.imread(str(dapi_pth), flags=cv2.IMREAD_UNCHANGED)

    row, col = dapi_img.shape
    print(row, col, rna_pth)
    
    out = []
    cell_info = zip(df_mn.cell_id, df_mn.vertex_y, df_mn.vertex_x)
    for (cid, r, c) in cell_info:
        # cid = cid.decode('utf-8')
        if not (sz <= r <= row - sz and sz <= c <= col - sz):
            print(f'{cid} is out of boundary, ignore')
            continue
        name = f'{r}_{c}_{cid}_{dapi_pth.stem}'
        dapi = dapi_img[r - sz: r + sz, c - sz: c + sz]
        rna = rna_coo[r - sz: r + sz, c - sz: c + sz, :gene_num].sum((0, 1)).todense()
        stat = [np.median(dapi), np.mean(dapi),
                np.min(dapi), np.max(dapi)]
        out.append((rna.tolist() +  stat))
    return out
        

def gamma_trans(gamma):
    gamma_table=[np.power(x/255,gamma)*255 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return gamma_table


def _random_color(size, num=128):
    h = np.random.rand(num)
    l = 0.4 + np.random.rand(num) / 5.0
    s = 0.5 + np.random.rand(num) / 2.0
    rl, gl, bl = list(), list(), list()
    for n in range(num):
        r, g, b = colorsys.hls_to_rgb(h[n], l[n], s[n])
        rl.append(max(int(255 * r), 100))
        gl.append(max(int(255 * g), 100))
        bl.append(max(int(255 * b), 100))
    rn = np.array(random.choices(rl, k=size))
    rn[0] = 0  # assign black to loc without cell_id
    gn = np.array(random.choices(gl, k=size))
    gn[0] = 0
    bn = np.array(random.choices(bl, k=size))
    bn[0] = 0
    return rn, gn, bn


def _check_overlap(img, pts):
    c_min, c_max = pts[:, 0].min(), pts[:, 0].max() + 1
    r_min, r_max = pts[:, 1].min(), pts[:, 1].max() + 1
    roi = img[r_min:r_max, c_min:c_max]
    msk = np.zeros_like(roi)
    pts[:, 0] -= c_min
    pts[:, 1] -= r_min
    cv2.fillPoly(msk, pts=[pts], color=1)
    # print(np.unique(msk))
    if not np.all(roi[msk == 1] == 0):
        # print(c_min, c_max, r_min, r_max)
        img[r_min:r_max, c_min:c_max] += msk * 100
    else:
        img[r_min:r_max, c_min:c_max] = msk * 100


def _roi_to_coord(h, w,
                  hei, wid,
                  roi, ovlp=200):
    crd = [h * roi, (h + 1) * roi,
           w * roi, (w + 1) * roi]
    crd = np.clip(crd,
                  [0, 0, 0, 0],
                  [hei, hei, wid, wid])

    crdo = [h * roi - ovlp, (h + 1) * roi + ovlp,
            w * roi - ovlp, (w + 1) * roi + ovlp]
    crdo = np.clip(crdo,
                   [0, 0, 0, 0],
                   [hei, hei, wid, wid])
    return crd.tolist(), crdo.tolist()


def prep_roi_img(img_pth, out_pth, roi, ovlp=200, gamma=0.5, coef=1):
    # crop-wise quantile median quantile
    # 32.0 473.5 4804.0 Lung 1
    # 60.0 494.0 4470.0 Lung 2
    extm = [32, 4804] if 'Lung1' in str(img_pth) else [60, 4470]
    print(extm)
    bdry = [32, 4804]
    img = imread(str(img_pth))

    is_rgb = len(img.shape) == 3
    hei, wid = img.shape[-2], img.shape[-1]
    h_num, w_num = ceil(hei / roi), ceil(wid / roi)
    print(h_num, w_num)
    for h in range(h_num):
        for w in range(w_num):
            print(h, w)
            crd, crdo = _roi_to_coord(h, w, hei, wid, roi, ovlp)
            print(crd, crdo, crd+crdo)
            if is_rgb:
                img_c = img[:, crdo[0]: crdo[1], crdo[2]: crdo[3]]
                img_c = img_c.transpose((1, 2, 0))
                img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)
            else:
                img_c = img[crdo[0]: crdo[1], crdo[2]: crdo[3]]

            img_c = np.clip(img_c.astype(np.float16), extm[0], extm[1])
            img_c = (img_c - bdry[0]) / (bdry[1] - bdry[0])
            img_c = (img_c * 255).astype(np.uint8)
            roi_nm = '_'.join(map(str, crd + crdo))
            cv2.imwrite(str(out_pth / f'{roi_nm}.jpg'), 
                        img_c)


def prep_roi_rna(df, masks, crd, crdo,
                 out_pth, rna_axs, color=None):
    roi_nm = '_'.join(map(str, crd + crdo))
    df = _df_to_roi(df, crdo, crdo, '{}_location')
    nucl_roi = masks[0][crdo[0]: crdo[1], crdo[2]: crdo[3]]
    cell_roi = masks[1][crdo[0]: crdo[1], crdo[2]: crdo[3]]

    print(roi_nm)
    if df.empty:
        assert np.all(nucl_roi == 0) and np.all(cell_roi == 0)
        print(f'ignore {roi_nm} that has no cells.')
        return

    df.feature_name = df.feature_name.map(rna_axs)
    df = df.groupby(list(df.columns), as_index=False).size()
    y = df.y_location.values
    x = df.x_location.values
    z = df.feature_name.values
    rna_coo = sparse.COO((y, x, z), df['size'].values,
                         shape=list(nucl_roi.shape) + [len(rna_axs)])

    if color is not None:
        img_n = color[0][nucl_roi]
        img_c = color[1][cell_roi]
        rna_np = rna_coo.sum(axis=-1).todense()
        img_rna = color[2][rna_np]

        cv2.imwrite(str(out_pth / f'{roi_nm}.jpg'),
                    np.stack([img_n, np.zeros_like(img_n), img_rna], axis=-1))
        # cv2.imwrite(str(out_pth / f'{roi_nm}_n.jpg'), img_n)
        # cv2.imwrite(str(out_pth / f'{roi_nm}_c.jpg'), img_c)
        # cv2.imwrite(str(out_pth / f'{roi_nm}_rna.jpg'), img_rna)
    else:
        msk_np = np.stack((nucl_roi, cell_roi), axis=-1)
        msk_coo = sparse.COO.from_numpy(msk_np)
        sparse.save_npz(str(out_pth / f'{roi_nm}_msk'), msk_coo)
        sparse.save_npz(str(out_pth / f'{roi_nm}_rna'), rna_coo)


def prep_crop(rna_pth, out_pth, df_mn, sz, debug=False, denum=10, gene_num=392):
    rna_coo = sparse.load_npz(str(rna_pth))
    msk_coo = sparse.load_npz(str(rna_pth).replace('_rna.npz', '_msk.npz'))

    dapi_pth = rna_pth.parent.parent / 'dapi' / (rna_pth.stem[:-4] + '.jpg')
    dapi_img = cv2.imread(str(dapi_pth), flags=cv2.IMREAD_UNCHANGED)
    # dapi_img = cv2.imread(str(hne_pth).replace('hne', 'dapi'),
    #                       flags=cv2.IMREAD_UNCHANGED)

    row, col, _ = rna_coo.shape
    print(row, col, rna_pth)

    if debug:
        # cell num < 1000000
        color = _random_color(1000000)
    dacc = 0
    cell_info = zip(df_mn.cell_id, df_mn.vertex_y, df_mn.vertex_x)
    for (cid, r, c) in cell_info:
        # cid = cid.decode('utf-8')
        if not (sz <= r <= row - sz and sz <= c <= col - sz):
            print(f'{cid} is out of boundary, ignore')
            continue
        name = f'{r}_{c}_{cid}_{dapi_pth.stem}'
        # hne = hne_img[r - sz: r + sz, c - sz: c + sz]
        dapi = dapi_img[r - sz: r + sz, c - sz: c + sz]
        nucl = msk_coo[r - sz: r + sz, c - sz: c + sz, 0]
        cell = msk_coo[r - sz: r + sz, c - sz: c + sz, 1]
        rna = rna_coo[r - sz: r + sz, c - sz: c + sz, :gene_num]
        if debug:
            img_n = color[0][nucl.todense()]
            # img_c = color[1][cell.todense()]
            img_c = np.zeros_like(img_n)
            rna_np = rna.sum(axis=-1).todense()
            img_rna = color[2][rna_np]
            # hne[:, :, 0] = img_n
            cv2.imwrite(str(out_pth / f'{name}_rna.jpg'),
                        np.stack([img_n, img_c, img_rna], axis=-1))

            # cv2.imwrite(str(out_pth / f'{name}_hne.jpg'), hne)
            cv2.imwrite(str(out_pth / f'{name}_dapi.jpg'), dapi)
            dacc += 1
            if dacc == denum:
                break
        else:
            # cv2.imwrite(str(out_pth / f'{name}_hne.png'), hne)
            cv2.imwrite(str(out_pth / f'{name}_dapi.png'), dapi)
            sparse.save_npz(str(out_pth / f'{name}_nucl'), nucl)
            sparse.save_npz(str(out_pth / f'{name}_cell'), cell)
            sparse.save_npz(str(out_pth / f'{name}_rna'), rna)


def sanity_check(pth):
    if not Path(pth.replace('rna.npz', 'cell.npz')).is_file():
        print('cell file {pth} missing')
    if not Path(pth.replace('rna.npz', 'nucl.npz')).is_file():
        print('nucl file {pth} missing')
    if not Path(pth.replace('rna.npz', 'hne.png')).is_file():
        print('hne file {pth} missing')
    if not Path(pth.replace('rna.npz', 'dapi.png')).is_file():
        print('dapi file {pth} missing')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path('Data/Xenium'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--img_type',
                        type=str,
                        choices=('hne', 'dapi'),
                        help='The image type (h&e or Dapi) to be processed')
    parser.add_argument('--roi_size',
                        type=int,
                        default=4000,
                        help='Size of cropped image region.')
    parser.add_argument('--roi_ovlp',
                        type=int,
                        default=200,
                        help='Overlap of cropped image region.')
    parser.add_argument('--cell_size',
                        type=int,
                        default=96,
                        help='Size of cropped single-cell image.')
    parser.add_argument('--prep_roi_img',
                        action='store_true',
                        help='crop image region with interest')
    parser.add_argument('--prep_roi_rna',
                        action='store_true',
                        help='Prepare roi of gene expr')
    parser.add_argument('--prep_crop',
                        action='store_true',
                        help='Prepare cropped cells training samples')
    parser.add_argument('--prep_stat',
                        choices=[None, 'all', 'subtype'],
                        help='Prepare cropped cells training samples')
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to visualize additional cell mask.')
    parser.add_argument('--calc_stat',
                        action='store_true',
                        help='Prepare cropped cells training samples')
    args = parser.parse_args()

    # slide-wise quantile 
    # [   0.    0.    0.   55. 1178. 2442. 7963.] Lung1
    # [   0.    0.    0.  254. 1462. 2340. 9362.] Lung2
    if args.prep_roi_img:
        for i in ('1', '2'):
            dat_pth = args.root / f'Lung{i}'
            img_pth = dat_pth / 'outs' / 'morphology_mip.ome.tif'
            # 7963 and 9362: max pixel of normal and tumor
            # 7963 / 9362 = 0.85
            coef = 0.75 if i == '1' else 1.

            out_pth = dat_pth / args.img_type
            out_pth.mkdir(parents=True, exist_ok=True)
            prep_roi_img(img_pth, out_pth,
                         args.roi_size, args.roi_ovlp,
                         coef=coef)

    if args.prep_roi_rna:
        rna_col = ['cell_id', 'y_location', 'x_location', 'feature_name']
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = list()
            for i in ('1', '2'):
                meta_pth = args.root / f'Lung{i}' / 'outs'
                if args.debug:
                    out_pth = args.root / f'Lung{i}' / 'debug_rna'
                else:
                    out_pth = args.root / f'Lung{i}' / 'rna'
                out_pth.mkdir(parents=True, exist_ok=True)
                with zarr.ZipStore(str(meta_pth / 'cells.zarr.zip'), mode='r') as cstore:
                    adata = sc.read_10x_h5(filename=str(meta_pth / 'cell_feature_matrix.h5'),
                                           gex_only=False)
                    rna_dct = dict(adata.var['feature_types'])
                    rna_axs = {k.encode('utf-8'): i for i,
                               k in enumerate(rna_dct)}
                    del adata

                    df_rna = pq.read_table(str(meta_pth / 'transcripts.parquet'),
                                           filters=[('cell_id', '!=', 'UNASSIGNED'),
                                                    ('qv', '>=', 20)],
                                           columns=rna_col).to_pandas()
                    # During the training, we use the labels stored in cell.parquet
                    # instead of the conflicted cell_id and overlaps* stored in transcripts.parquet
                    df_rna = df_rna.drop(columns=['cell_id'])
                    _um_to_pixel(df_rna, '{}_location')

                    masks = zarr.group(store=cstore, overwrite=False).masks
                    hei, wid = masks[0].shape[-2], masks[0].shape[-1]
                    h_num = ceil(hei / args.roi_size)
                    w_num = ceil(wid / args.roi_size)
                    print(h_num, w_num)

                    color = None
                    if args.debug:
                        # cell num < 1000000
                        color = _random_color(1000000)

                    for h in range(h_num):
                        for w in range(w_num):
                            print(h, w)
                            crd, crdo = _roi_to_coord(h, w, hei, wid,
                                                      args.roi_size, args.roi_ovlp)
                            prep_args.append([df_rna, masks, crd, crdo,
                                              out_pth, rna_axs, color])
            pool.starmap(prep_roi_rna, prep_args)

    if args.prep_crop:
        cell_col = ['cell_id', 'y_centroid', 'x_centroid',
                    'transcript_counts', 'control_probe_counts',
                    'control_codeword_counts', 'total_counts']
        type_col = {'vertex_y': int, 'vertex_x': int}
        for rep in ('Lung1', 'Lung2'):
            with multiprocessing.Pool(processes=args.core) as pool:
                prep_args = list()
                if args.debug:
                    out_pth = args.root / rep / 'debug_crop'
                else:
                    out_pth = args.root / 'GAN' / 'crop' / rep
                out_pth.mkdir(parents=True, exist_ok=True)

                meta_pth = args.root / rep / 'outs'
                # df_nmsk.cell_id.is_monotonic_increasing True
                df_nmsk = pq.read_table(str(meta_pth / 'nucleus_boundaries.parquet'),
                                        columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()
                _um_to_pixel(df_nmsk, 'vertex_{}')
                # The first and last pt for each cells (>= 7 boundary pts) are identical
                # df_nmsk.groupby(['cell_id']).first().equals(df_nmsk.groupby(['cell_id']).last()))
                # df_nmsk.duplicated assign False to the last pt of a cell,
                df_last = df_nmsk[df_nmsk.duplicated(subset=['cell_id'],
                                                     keep='last')]
                # calc the centroid of each cell based on nucleus boundary
                df_mean = df_last.groupby(['cell_id'],
                                          as_index=False)[['vertex_y', 'vertex_x']].mean().astype(type_col)

                if args.prep_stat == 'subtype':
                    # only epithelial cell
                    df = pd.read_csv('Data/Xenium/GAN/crop/metadata.csv')
                    dfs = df[(df['slide_ID_numeric'] == int(rep[-1])) & (df['kmeans_2_clusters'] == 2)]
                roi_pth = args.root / rep / 'rna'
                roi_lst = list(roi_pth.glob('*rna.npz'))
                cell_acc = 0
                for rid, roi in enumerate(roi_lst):
                    crd_all = list(map(int, roi.stem.split('_')[:-1]))
                    crd, crdo = crd_all[:4], crd_all[4:]

                    df_mn = _df_to_roi(df_mean, crd, crdo, 'vertex_{}')
                    df_mn.cell_id = df_mn.cell_id.str.decode('utf-8')
                    roi_len = len(df_mn)
                    if args.prep_stat == 'subtype':
                        df_mn = df_mn[df_mn.cell_id.isin(dfs.cell_id)]
                        print(rid, roi_len, len(df_mn))
                    prep_args.append([roi, out_pth, df_mn,
                                      args.cell_size // 2, args.debug])
                    cell_acc += len(df_mn)
                if args.prep_stat is None:
                    assert cell_acc == len(df_mean)
                if args.prep_stat is None:
                    pool.starmap(prep_crop, prep_args)
                else:
                    out = pool.starmap(prep_crop1, prep_args)
                    out = np.array(sum(out, []))
                    print(out.shape)
                    suffix = '' if args.prep_stat == 'all' else '_epith'
                    out = np.save(f'stats/0_xenium_{rep[-1]}{suffix}.npy', out)

    # with multiprocessing.Pool(processes=args.core) as pool:
    #     prep_args = list()
    #     for rep in ('Rep1', 'Rep2'):
    #         out_pth = args.root / 'GAN' / 'crop' / rep
    #         for gene in out_pth.rglob('*_rna.npz'):
    #             prep_args.append((str(gene), ))
    #         pool.starmap(sanity_check, prep_args)

    if args.calc_stat:
        with open('Data/Xenium/GAN/crop/transcripts.pickle', 'rb') as fp:
            gene_name = pickle.load(fp)
            for suffix in ('', '_epith'):
                for i in (1, 2):
                    arr = np.load(f'stats/0_xenium_{i}{suffix}.npy')
                    print(np.min(arr[:, 394]),
                          np.quantile(arr[:, 394], 0.05),
                          np.median(arr[:, 392]),
                          np.quantile(arr[:, 395], 0.95),
                          np.max(arr[:, 395]))
                    
                    for gene in ('MUC1', 'KRT7', 'EPCAM'):
                        gid = gene_name.index(gene)
                        for chn, stt in {392:'median', 395:'max'}.items():
                            df = pd.DataFrame({gene: arr[:, gid], stt:arr[:, chn]})
                            print(gene, stt)
                            sns.lmplot(data=df, x=gene, y=stt, 
                                    scatter_kws={'s': 0.5, 'alpha':0.1})
                            plt.title(i)
                            plt.savefig(f'plot_xenium/xenium{suffix}_{i}_{gene}_{stt}.png')
                            plt.close()
