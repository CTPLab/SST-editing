import sparse
import pickle
import tiledb
import tiledbsoma
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from cleanfid import fid
import pyarrow.parquet as pq

# Compared to 0.2125, 4.7058825 cause smaller inconsistency
# between transcripts.parquet and cells.zarr.zip
PX_SZ = 4.7058825
pd.options.mode.chained_assignment = None


def _um_to_pixel(df, crd_nm):
    for axis in ('x', 'y'):
        # astype(float) could slightly improve the inconsistency on dgx
        df[crd_nm.format(axis)] = df[crd_nm.format(axis)].astype(float) * PX_SZ
        # this gives us the least error rate compared to use round()
        df[crd_nm.format(axis)] = df[crd_nm.format(axis)].astype(int)


def _df_to_roi(df, roi_crop, roi_trns,
               crd_nm, crop=True):
    if crop:
        is_roi_h = (df[crd_nm.format('y')] >= roi_crop[0]) & \
            (df[crd_nm.format('y')] < roi_crop[1])
        is_roi_w = (df[crd_nm.format('x')] >= roi_crop[2]) & \
            (df[crd_nm.format('x')] < roi_crop[3])
        df = df[(is_roi_h) & (is_roi_w)]
    # covert the global- to local- coordinate for the roi image
    # hei_st = crd[0], wid_st = crd[2]
    df[crd_nm.format('y')] -= roi_trns[0]
    df[crd_nm.format('x')] -= roi_trns[2]
    return df


def prep_Visium_meta(pth=Path('Data/Visium')):
    df = pd.read_csv(str(pth / '10x_tap' / '10x_image-paths_resolution.csv'))
    print(df)
    img_dir = pth / 'GAN' / 'crop'
    img_pth = list(img_dir.rglob('*.png'))
    img_pth = [str(pth).replace(str(img_dir)+'/', '') for pth in img_pth]
    dfm = pd.DataFrame(img_pth, columns=['path', ])
    dfm['replicate'] = dfm.path.map(lambda x: Path(x).parent.name)

    pat_dct = dict(zip(df.replicate, df['sample']))
    dfm['patient'] = dfm.replicate.map(pat_dct)

    sld_dct = dict(zip(df.replicate, df['image_name']))
    dfm['slide_name'] = dfm.replicate.map(sld_dct)

    res_dct = dict(zip(df.replicate, df['resolution']))
    dfm['resolution'] = dfm.replicate.map(res_dct)

    imm_dct = dict(zip(df.replicate, df['immune_phenotype']))
    dfm['immune_phenotype'] = dfm.replicate.map(imm_dct)

    dfm.to_csv(str(pth / 'GAN' / 'crop' / 'metadata.csv'),
               index=False)


def align_visium_meta(pth=Path('Data/Visium/GAN/crop')):
    with open(str(pth / 'transcripts.pickle'), 'rb') as f:
        gene_lst = pickle.load(f)
        df = pd.DataFrame(columns=gene_lst)
        npz_pth = pd.read_csv(str(pth / 'metadata.csv')).path
        for nid, npth in enumerate(npz_pth):
            row = sparse.load_npz(str(pth / npth.replace('hne.png', 'rna.npz')))
            df.loc[npth] = row.todense()
            if (nid + 1) % 5000 == 0:
                print(nid + 1)
        df = df.reset_index(names='path')
        df.to_csv(str(pth / 'metadata_cell.csv'), index=False)
        df.to_csv(str(pth / 'metadata_img.csv'), index=False)


def prep_CosMx_path(row, root):
    cell_name = f'{row.CenterY_local_px}_{row.CenterX_local_px}_{row.cell_ID}_F{str(row.fov).zfill(3)}_flo.png'
    cell_pth = root / row.tissue / cell_name
    return str(Path(row.tissue) / cell_name), cell_pth.is_file()


def prep_CosMx_meta(in_pth=Path('Data/CosMx'),
                    out_pth=Path('Data/CosMx/GAN'),
                    columns=['fov', 'y_FOV_px', 'x_FOV_px',
                             'slide_ID_numeric', 'cellType', 'niche']):
    # prepare the meta data
    config = tiledb.Config()
    ctx = tiledb.Ctx(config)
    pySoma = tiledbsoma.SOMACollection(str(in_pth / 'raw' / 'LiverDataRelease'),
                                       ctx=ctx)
    df = pySoma['RNA'].obs.df(attrs=columns)
    df.reset_index(inplace=True)
    df = df.rename(columns={'obs_id': 'path'})
    print(df.head(), len(df))
    flo_pth = (out_pth / 'crop').rglob('*_flo.png')
    flo_dct = {flo.stem[:-4]: str(Path(flo.parent.name) / flo.name)
               for flo in flo_pth}
    print(dict(list(flo_dct.items())[:10]))
    df = df[df.path.isin(list(flo_dct.keys()))]
    print(len(df))
    df.path = df.path.map(flo_dct)
    print(df.head())
    df.to_csv(str(out_pth / 'crop' / 'metadata.csv'), index=False)


def align_CosMx_meta(pth=Path('Data/CosMx/')):
    meta_pth = pth / 'GAN' / 'crop'
    dfm = pd.read_csv(str(meta_pth / 'metadata.csv'))
    cid = dfm['path']
    cid_expr = dfm['path'].map(
        lambda x: x.split('/')[-1].replace('_flo.png', ''))
    cid_dct = dict(zip(cid_expr, cid))
    del dfm
    df1 = pd.read_csv(str(pth / 'Liver1' / 'exprMat_file.csv'), 
                      index_col=0).astype(np.int16)
    df2 = pd.read_csv(str(pth / 'Liver2' / 'exprMat_file.csv'), 
                      index_col=0).astype(np.int16)
    dfe = pd.concat([df1, df2])
    del df1, df2
    dfe = dfe.reindex(cid_expr)
    dfe.index = dfe.index.map(cid_dct)
    print(dfe.head())
    dfe.to_csv(str(meta_pth / 'metadata_cell.csv'))


def prep_Xenium_meta(img_ext='*_dapi.png',
                     col_nam=['path', 'cell_id',
                              'local_y', 'local_x', 'slide_ID_numeric', ],
                     in_pth=Path('Data/Xenium'),
                     out_pth=Path('Data/Xenium/GAN')):
    cnt_lst = ['transcript_counts',  # 0 - 279
               'control_probe_counts',  # 280 - 299
               'control_codeword_counts',  # 300 - 340
               # blank codeword 341 - 540
               'total_counts']
    # prepare the meta data
    df = list()
    for r in ('Lung1', 'Lung2'):
        df_cell = pq.read_table(str(in_pth / r / 'outs' / 'cells.parquet'),
                                columns=['cell_id', ] + cnt_lst).to_pandas()

        img_pth = (out_pth / 'crop' / r).glob(img_ext)
        img_pth = [(f'{i.parent.name}/{i.name}',  # path
                    i.name.split('_')[2],  # cell_id
                    int(i.name.split('_')[0]),  # local_y
                    int(i.name.split('_')[1]),  # local_x
                    int(i.parent.name[-1]), )
                   for i in img_pth]
        df_img = pd.DataFrame(img_pth, columns=col_nam)
        for cnt in cnt_lst:
            cnt_dct = dict(zip(df_cell.cell_id.str.decode('utf-8'), df_cell[cnt]))
            df_img[cnt] = df_img.cell_id.map(cnt_dct)
        df.append(df_img)
        print(len(df[-1]))

        for clt in range(11):
            if clt == 0:
                clt_nam = 'graphclust'
            if clt == 1:
                continue
            if clt > 1:
                clt_nam = f'kmeans_{clt}_clusters'
            print(clt_nam)

            clt_pth = in_pth / r / 'outs' / 'analysis' / 'clustering' / \
                f'gene_expression_{clt_nam}'
            df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
            print(len(df_raw))
            # bar_max = df_raw.Barcode.max()
            # assert df_raw.Barcode.is_monotonic_increasing
            # df_raw = df_raw.set_index('Barcode').reindex(range(1, bar_max + 1),
            #                                              fill_value=0)
            # df_raw = df_raw.reset_index().astype(int)
            raw_dct = dict(zip(df_raw.Barcode, df_raw.Cluster))
            df[-1][clt_nam] = df[-1].cell_id.map(raw_dct)
    df = pd.concat(df, ignore_index=True)
    df_meta = df[df.graphclust.notna()]
    # convert to int since 9th col (graphclust)
    df_meta.iloc[:, 9:] = df_meta.iloc[:, 9:].astype(int)
    assert not df_meta.isnull().values.any()
    df_meta.to_csv(str(out_pth / 'crop' / 'metadata.csv'), index=False)
    df[~df.graphclust.notna()].to_csv(str(out_pth / 'crop' / 'metadata_no.csv'), index=False)


def align_xenium_meta(pth=Path('Data/Xenium/')):
    cmp_dct = {'Gene Expression': 'transcript_counts',
               'Negative Control Codeword': 'control_codeword_counts',
               'Negative Control Probe': 'control_probe_counts',
               # this is not very useful, only meant for
               # simply assigning columns name for pd.read_table
               'total_counts': 'total_counts'}

    fill_val = {'transcript_counts': 0, 'control_codeword_counts': 0, 
                'control_probe_counts':0, 'Unassigned Codeword': 0}
    type_col = {'transcript_counts': int, 'control_codeword_counts': int, 
                'control_probe_counts':int, 'Unassigned Codeword': int}


    dfm = pd.read_csv(str(pth / 'GAN/crop/metadata.csv'))
    cid = dfm['path']
    # Rep1/124
    cid_expr = dfm['path'].map(lambda x: x.split('/')[0] + x.split('_')[2])
    cid_dct = dict(zip(cid_expr, cid))

    dfe, dfc = [], []
    # the core impl comes from test_Xenium_meta.py
    for i in ('Lung1', 'Lung2'):
        meta_pth = pth / i / 'outs'
        adata = sc.read_10x_h5(filename=str(meta_pth / 'cell_feature_matrix.h5'),
                               gex_only=True)

        df_cell = pq.read_table(str(meta_pth / 'cells.parquet'),
                                columns=['cell_id'] + list(cmp_dct.values())).to_pandas()
        dc = df_cell.copy().set_index('cell_id')
        dc.index = dc.index.str.decode('utf-8')
        dc.index = dc.index.map(lambda x: f'{i}{x}')
        dfc.append(dc)
        del dc
        # copy the cells.parquet to the scanpy data
        adata.obs = df_cell.copy()

        gene_dct = dict(adata.var['feature_types'])
        df_cell.drop(df_cell[df_cell['total_counts'] == 0].index,
                     inplace=True)
        if not (pth / i / 'exprMat_file.csv').is_file():
            df_gene = pq.read_table(str(meta_pth / 'transcripts.parquet'),
                                    filters=[('cell_id', '!=', 'UNASSIGNED'),
                                             # https://www.10xgenomics.com/cn/resources/analysis-guides/performing-3d-nucleus-segmentation-with-cellpose-and-generating-a-feature-cell-matrix
                                             # critical QV thres >= 20
                                             ('qv', '>=', 20)],
                                    columns=['cell_id', 'feature_name']).to_pandas()

            # the following steps can be very slow
            print('start replacing the gene_dct')
            # convert binary gene name to string then to feature_types
            # for the following count comparison (~3000s)
            df_gene.feature_name = df_gene.feature_name.\
                str.decode('utf-8')

            print('start counting the exprs')
            # compute the counts for Gene expr, neg ctr prob,
            # neg ctr cod and blank code (can be slow)
            df_gene = df_gene.groupby(['cell_id', 'feature_name'],
                                      as_index=False).size()

            print('start transposing the df')
            # transpose the df format so it is similar to df_cell
            df_gene = df_gene.pivot(index='cell_id', columns='feature_name',
                                    values='size').reset_index()
            df_gene.columns.name = None
            df_gene.cell_id = df_gene.cell_id.str.decode('utf-8')
            df_gene = df_gene.set_index('cell_id')
            df_gene = df_gene.fillna(0).astype(int)
            df_gene_col = df_gene.columns.isin(list(gene_dct.keys()))
            df_gene = df_gene.loc[:, df_gene_col]
            print(df_gene.head())
            print(len(df_gene))
            df_gene.to_csv(str(pth / i / 'exprMat_file.csv'))
        else:
            df_gene = pd.read_csv(str(pth / i / 'exprMat_file.csv'),
                                  index_col='cell_id').astype(np.int16)
            print(len(df_gene), len(df_cell))
        with open(str(pth / 'GAN/crop/transcripts.pickle'), 'rb') as f:
            gene_lst = pickle.load(f)
            assert df_gene.columns.values.tolist() == gene_lst
        assert (df_gene.iloc[:, :].sum(axis=1).values ==
                df_cell['transcript_counts'].values).all()

        df_gene.index = df_gene.index.map(lambda x: f'{i}{x}')
        print(df_gene.head())
        dfe.append(df_gene)
        del df_gene

    dfc = pd.concat(dfc)
    dfc = dfc.reindex(cid_expr)
    print(len(dfc))

    # Here, we let metadata_cell.csv align with cropped images
    # and its metadata.csv, because we preprocess the cropped images first
    dfe = pd.concat(dfe)
    dfe = dfe.reindex(cid_expr).fillna(0).astype(int)
    print(dfe.iloc[:, :].sum(axis=1).values)
    assert (dfe.iloc[:, :].sum(axis=1).values ==
            dfc['transcript_counts'].values).all()
    dfe.index = dfe.index.map(cid_dct)
    print(dfe.head())
    print(len(dfe))
    dfe.astype(int).to_csv(str(pth / 'GAN/crop/metadata_cell.csv'))


def save_transcript_list(data):
    print(data)
    if data == 'Visium':
        lst = ['SLC2A1', 'CCN1', 'ATP1A1', 'S100A1', 'NES', 'SLC4A5', 'PAX3', 'MLPH', 'SEMA3B', 'WNT5A',
               'MITF', 'ROPN1B', 'SLIT2', 'SLC45A2', 'TGFBI', 'GFRA3', 'PDGFRB', 'ABCB5', 'AQP1', 'EGFR',
               'TMEM176B', 'GFRA2', 'LOXL2', 'MLANA', 'TYRP1', 'TNC', 'VIM', 'LOXL4', 'PLEKHB1', 'RAB38',
               'TYR', 'SLC2A3', 'PMEL', 'CDK2', 'ERBB3', 'NT5DC3', 'POSTN', 'SLC22A17', 'SERPINA3', 'AKT1',
               'CAPN3', 'CDH1', 'CDH13', 'NGFR', 'SOX9', 'CDH2', 'TCF4', 'BCL2', 'CDH19', 'MBP', 'MIA',
               'AXL', 'BIRC7', 'S100B', 'PRAME', 'SOX10', 'GPR143', 'GPM6B', 'PIR', 'GJB1', 'BGN']
    elif data == 'Xenium':
        adata = sc.read_10x_h5(filename=f'Data/{data}/Lung1/outs/cell_feature_matrix.h5',
                               gex_only=True)
        gene_dct = dict(adata.var['feature_types'])
        lst = [k for k in gene_dct]

        adata2 = sc.read_10x_h5(filename=f'Data/{data}/Lung2/outs/cell_feature_matrix.h5',
                                gex_only=True)
        gene_dct2 = dict(adata2.var['feature_types'])
        assert lst == list(gene_dct2.keys())
    elif data == 'CosMx':
        df_expr = pd.read_csv(f'Data/{data}/Liver1/exprMat_file.csv',
                              index_col=0)
        lst = [e for e in df_expr.columns]

        df_expr2 = pd.read_csv(f'Data/{data}/Liver2/exprMat_file.csv',
                               index_col=0)
        assert lst == df_expr2.columns.values.tolist()
    else:
        raise NameError('unrecognized data name {data}')

    with open(f'Data/{data}/GAN/crop/transcripts.pickle', 'wb') as fp:
        pickle.dump(lst, fp)


def save_fid_stats(data, model_name):
    # need to modify the get_folder_features function in fid.py of clean-FID package
    # if 'CosMx' in fdir:
    #     files = sorted([f for f in glob(os.path.join(fdir, f"**/*_flo.png"), recursive=True)])
    # elif 'Xenium' in fdir:
    #     files = sorted([f for f in glob(os.path.join(fdir, f"**/*_hne.png"), recursive=True)])
    # elif 'Visium' in fdir:
    #     files = sorted([f for f in glob(os.path.join(fdir, f"**/*.png"), recursive=True)])
    fid.make_custom_stats(data, f'Data/{data}/GAN/crop',
                          mode='clean',
                          model_name=model_name,
                          num_workers=8)

def proc_xenium_cell_id():
    df = pd.read_csv('Data/Xenium/GAN/crop/metadata_img.csv',
                     index_col=0)
    dfl = []
    for i in range(2):
        dfm = pq.read_table(f'Data/Xenium/Lung{i + 1}/outs/nucleus_boundaries.parquet',
                            columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()
        
        # calc the centroid of each cell based on nucleus boundary
        # minimum 7 vertices
        dfm = dfm.groupby(['cell_id'],
                          as_index=False)[['vertex_y', 'vertex_x']].mean()
        _dct = dict(zip(dfm.cell_id.str.decode('utf-8'), dfm.index.values + 1))
        cnd = lambda x: x.replace(Path(x).name.split('_')[2], 
                                  str(_dct[Path(x).name.split('_')[2]]))
        df1 = df[df.index.str.contains(f'Lung{i + 1}')]
        df1.index = df1.index.map(cnd)
        dfl.append(df1)
        del df1
    del df
    df = pd.concat(dfl)
    print(df)
    df.to_csv('Data/Xenium/GAN/crop/metadata_bar.csv')

if __name__ == '__main__':
    # for data in ['Xenium', ]:
    #     save_transcript_list(data)

    # # prep_Visium_meta()
    # # prep_CosMx_meta()
    # prep_Xenium_meta()
    # # align_visium_meta()
    # # align_CosMx_meta()
    # align_xenium_meta()

    proc_xenium_cell_id()
    

