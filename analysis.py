import os
import sys
import torch
import pickle
import random
import logging
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from cleanfid.features import build_feature_extractor
from wilds.common.data_loaders import get_eval_loader, get_train_loader

from args import parse_args
from Dataset.STDataset import STDataset
from style3.inversion.models.psp3 import pSp
from style3.models.stylegan3.model import Decoder
from style3.inversion.options.train_options import TrainOptions

from utils.plot import edit_plot
from utils.edit import run_demo_edit, edit_gene, run_fov_edit
from utils.metric import run_metric, get_eigen, calc_img_dist, calc_img_reco, run_gene_counts

sys.path.append('.')


def setup_log(args):
    r"""
    Configure the logging document that records the
    critical information during evaluation

    Args:
        args: Arguments that are implemented in args.py file
              such as data_name, data_splt.
    """

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(str(args.save_path / f'log_{args.task}_{args.data_splt}'),
                                        mode='w'))
    logging.basicConfig(level=logging.INFO,
                        format=head,
                        style='{', handlers=handlers)
    logging.info(f'Start with arguments {args}')


def setup_seed(seed):
    r"""

    Args:
        seed: Seed for reproducible randomization.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(args, ckpt_pth, is_GAN=False):
    r"""
    Initialize the model and then load the weights 

    Args:
        args: Arguments that are implemented in args.py file
              such as data_name, data_splt.
        ckpt_pth: Path to the checkpoint
        is_GAN: Load GAN model if True else load GAN Inversion model
    """

    if is_GAN:
        model = Decoder(args.decoder, args, ckpt_pth).decoder
        for layer_idx in range(model.num_layers):
            res = (layer_idx + 5) // 2
            shape = [args.n_eval, 1, 2 ** res, 2 ** res]
            setattr(model.noises, f'noise_{layer_idx}',
                    torch.randn(*shape))
        print(f'load {args.decoder} done!')
    else:
        if args.decoder == 'style3':
            raise NotImplementedError()
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        opts = ckpt['opts']
        opts.update({'checkpoint_path': ckpt_pth})
        opts = TrainOptions(**opts)
        opts.n_eval = args.n_eval
        model = pSp(opts)
        print(f'load {args.decoder} inversion done!')

    model.eval().cuda()
    return model


def get_mfeat():
    r"""
    Load the feature extractor for calc d_FID

    """

    mfeat = {'clean': build_feature_extractor('clean')}
    return mfeat


def get_loader(args,
               data,
               cond=None):
    r"""
    Initialize the dataset loader

    Args:
        data: Name of the ST platform
        ckpt_pth: Path to the checkpoint
        cond: Label for the subset to be loaded
    """

    subset = data if cond is None else data.get_subset(cond)

    if 'GAN' in args.analysis and args.task != 'fov_demo':
        data_loader = get_train_loader
    else:
        data_loader = get_eval_loader

    drop_last = False
    if args.task == 'fov_demo':
        drop_last = True

    dload = data_loader('standard', subset, args.size_bat,
                        **{'drop_last': drop_last})
    return dload


def main(args):
    r"""
    Main function of generating the results reported in the paper
    """

    root = Path(f'Data/{args.data_name}/GAN/crop')
    if args.gene_name:
        # Get the indices based on the list of gene names
        # this is meant for targeted gene expression editing
        with open(str(root / 'transcripts.pickle'), 'rb') as fp:
            gene_all = pickle.load(fp)
            gene_idx = [gene_all.index(g) for g in args.gene_name.split(',')]
            print(args.data_name, gene_idx)
            gene_nm = f'_{args.gene_name.replace(",", "_")}'
    else:
        gene_idx, gene_nm = None, ''

    dfm = pd.read_csv(str(root / 'metadata.csv'), index_col=0)
    subnm, subst = '', None
    if args.fov or args.task == 'fov_demo':
        if args.data_name == 'CosMx':
            fov_id = ['097', '055']
            bdry = [(2128 - 1024, 2128 + 1024), (2128 - 1024, 2128 + 1024)]
            sub1 = (dfm['slide_ID_numeric'] == 1) & \
                (dfm['fov'] == int(fov_id[0]))
            sub2 = (dfm['slide_ID_numeric'] == 2) & \
                (dfm['fov'] == int(fov_id[1]))
            suby = (dfm['y_FOV_px'] >= bdry[0][0]) & \
                (dfm['y_FOV_px'] < bdry[0][1])
            subx = (dfm['x_FOV_px'] >= bdry[0][0]) & \
                (dfm['x_FOV_px'] < bdry[0][1])
            subst = (sub1 | sub2) & (suby & subx)
        elif args.data_name == 'Xenium':
            fov_id = ['16000_20000_4000_8000_15800_20200_3800_8200',
                      '12000_16000_32000_36000_11800_16200_31800_36200']
            bdry = [(1900, 1900 + 2048), (2200 - 1024, 2200 + 1024)]
            sub1 = (dfm['slide_ID_numeric'] == 1) & \
                (dfm.index.str.contains(fov_id[0]))
            sub1y = (dfm['local_y'] >= bdry[0][0]) & \
                    (dfm['local_y'] < bdry[0][1])
            sub1x = (dfm['local_x'] >= bdry[0][0]) & \
                    (dfm['local_x'] < bdry[0][1])
            sub2 = (dfm['slide_ID_numeric'] == 2) & \
                (dfm.index.str.contains(fov_id[1]))
            sub2y = (dfm['local_y'] >= bdry[1][0]) & \
                    (dfm['local_y'] < bdry[1][1])
            sub2x = (dfm['local_x'] >= bdry[1][0]) & \
                    (dfm['local_x'] < bdry[1][1])
            subst = (sub1 & sub1y & sub1x) | (sub2 & sub2y & sub2x)

    if args.subtype:
        if args.data_name == 'CosMx':
            subnm = '_cellType'
            sub1 = (dfm['slide_ID_numeric'] == 1) & \
                (dfm['cellType'].str.contains('Hep.'))
            sub2 = (dfm['slide_ID_numeric'] == 2) & \
                (dfm['cellType'].str.contains('tumor_'))
            subst = sub1 | sub2 if subst is None else subst & (sub1 | sub2)
        elif args.data_name == 'Xenium':
            subnm = '_kmeans_2_clusters'
            subst = dfm['kmeans_2_clusters'] == 2 if subst is None else \
                subst & (dfm['kmeans_2_clusters'] == 2)

    if args.analysis == 'RAW':
        stat_pth = args.save_path / args.task
        stat_pth.mkdir(parents=True, exist_ok=True)

        if args.task == 'gene_eig':
            # Generate the transformed gene expressions of one cell (sub)population
            # such that its sample covariance matrix matches another
            dfg = pd.read_csv(str(root / 'metadata_img.csv'), index_col=0)
            if subst is not None:
                dfm, dfg = dfm[subst], dfg[subst]
                assert dfm.index.equals(dfg.index)
            dfm = dfm[[args.data_splt, ]].values
            dfm = np.squeeze(dfm)
            spl, scn = np.unique(dfm, return_counts=True)
            dfg = dfg.astype(np.int16).to_numpy()
            print(spl, scn, dfm.shape, dfg.shape)
            epth = f'stats/{args.data_name}{subnm}_{args.data_splt}_'.lower()
            for s in spl:
                if Path(f'{epth}{s}_eig.pt').is_file():
                    continue
                df = torch.from_numpy(dfg[dfm == s, ])
                print(s, df.shape)
                eig = get_eigen((df.double().T @ df.double()) / df.shape[0])
                # Save the unchanged eigenvalue, eigenvector of
                # compared gene expressions
                torch.save(eig, f'{epth}{s}_eig.pt')

            # Paired eigen dct
            edct = {'1': '2', '2': '1'}
            with open(str(root / 'transcripts.pickle'), 'rb') as fp:
                gene_name = pickle.load(fp)
            df = []
            for s in spl:
                if s == 0 and args.data_name == 'Xenium':
                    continue
                sub = torch.from_numpy(dfg[dfm == s, ]).float()
                df.append(sub.cuda())
                eigd = torch.load(f'{epth}{s}_eig.pt')
                eigr = torch.load(f'{epth}{edct[str(s)]}_eig.pt')
                # The gene expression editing function
                dfe = edit_gene(df[-1], eigd, eigr)
                dif_mean = (df[-1] - dfe).mean(0)
                dif_std = (df[-1] - dfe).std(0)
                gene_idx = torch.argsort(dif_mean.abs(), descending=True)
                gene_lst = [(gene_name[i],
                             f'{float(dif_mean[i].cpu().numpy()):.2f}',
                             f'{float(dif_std[i].cpu().numpy()):.2f}')
                            for i in gene_idx]
                print(gene_lst[:50], '\n')
            df_mean = df[0].mean(0) - df[1].mean(0)
            df_avg0 = df[0].mean(0)
            df_std0, df_std1 = df[0].std(0), df[1].std(0)
            gene_idx = torch.argsort(df_mean.abs(), descending=True)
            gene_lst = [(gene_name[i],
                         f'{float(df_mean[i].cpu().numpy()):.2f}',
                         f'{float(df_avg0[i].cpu().numpy()):.2f}',
                         f'{float(df_std0[i].cpu().numpy()):.2f}',
                         f'{float(df_std1[i].cpu().numpy()):.2f}')
                        for i in gene_idx]
            print(gene_lst[:50], '\n')
        elif args.task == 'gene_plot':
            if args.data_name == 'CosMx':
                gene_sub = ['HLA-A', 'B2M', 'APOA1', 'TTR', 'MALAT1']
            elif args.data_name == 'Xenium':
                gene_sub = ['MUC1', 'KRT7', 'RBM3', 'EPCAM', 'TOMM7']

            dfg = pd.read_csv(str(root / 'metadata_img.csv'), index_col=0)
            if subst is not None:
                dfm, dfg = dfm[subst], dfg[subst]
            dfm = dfm[[args.data_splt, ]].values
            dfm = np.squeeze(dfm)
            dfg = dfg.astype(np.int16).to_numpy()
            epth = f'stats/{args.data_name}{subnm}_{args.data_splt}_'.lower()
            with open(str(root / 'transcripts.pickle'), 'rb') as fp:
                gene_name = pickle.load(fp)
            eig0 = torch.load(f'{epth}1_eig.pt')
            eig1 = torch.load(f'{epth}2_eig.pt')
            gene0, gene1 = dfg[dfm == 1, ], dfg[dfm == 2, ]
            edit0 = edit_gene(torch.from_numpy(gene0).float(),
                              eig0, eig1)
            edit1 = edit_gene(torch.from_numpy(gene1).float(),
                              eig1, eig0)
            print(gene0.shape, gene1.shape, edit0.shape, edit1.shape)

            edit_plot(stat_pth, gene0, gene1,
                      edit0.numpy(), edit1.numpy(),
                      gene_name,
                      gene_sub,
                      subnm)

    elif 'GAN' in args.analysis:
        data = STDataset(args.data_name, args.gene_num, args.gene_spa,
                         split_scheme=args.data_splt, flter_subset=subst)
        assert args.data_splt is not None
        is_edit = False if args.analysis == 'GANIM' else True

        # paired eigen dct
        edct = {'1': '2', '2': '1'}
        epth = f'stats/{args.data_name}{subnm}_{args.data_splt}_'.lower()

        i_iter = list(range(50000, 850000, 50000)) if args.n_iter is None \
            else [args.n_iter, ]
        for i in i_iter:
            model_pth = args.ckpt_path / f'{str(i).zfill(6)}.pt'
            if not model_pth.is_file():
                print(f'{model_pth} is not a pt file, ignore')
                continue
            model = get_model(args, str(model_pth),
                              is_GAN=args.analysis == 'GAN')
            for cond, cval in data._cond_dct.items():
                print(cond, cval)
                eigen = [torch.load(f'{epth}{cond}_eig.pt'),
                         torch.load(f'{epth}{edct[str(cond)]}_eig.pt')]
                dload = get_loader(args, data, cond)
                print(eigen[0][0].shape, eigen[0][1].shape,
                      eigen[1][0].shape, eigen[1][1].shape)
                stat_pth = args.save_path / f'{cond}{subnm}_{i}{gene_nm}'
                stat_pth.mkdir(parents=True, exist_ok=True)
                if args.task == 'metric':
                    run_metric(args, stat_pth, dload, eigen,
                               model, get_mfeat(),
                               total=min(50000, cval) if is_edit else cval,
                               repeat=4 if is_edit else 1,
                               gene_idx=gene_idx)
                    if is_edit:
                        refs = dict()
                        for nm in ('clean',):
                            if cond is not None:
                                refs[nm] = [torch.load(f'{epth}{cond}_{nm}.pt'),
                                            torch.load(f'{epth}{edct[str(cond)]}_{nm}.pt')]
                            else:
                                refs[nm] = [torch.load(f'{epth}{nm}.pt'),
                                            torch.load(f'{epth}{nm}.pt')]
                        calc_img_dist(stat_pth, refs)
                    else:
                        calc_img_reco(stat_pth)
                elif args.task == 'demo':
                    run_demo_edit(args, stat_pth,
                                  dload, eigen, model,
                                  gene_idx=gene_idx)
                elif args.task == 'fov_demo':
                    # Create the video demo for the GAN Inversion model
                    if args.data_name == 'CosMx':
                        fov_img = Image.open(
                            f'Data/CosMx/Liver{cond}/CellComposite/CellComposite_F{fov_id[cond -1]}.jpg')
                        cat_img = Image.open(
                            f'Experiment/CosMx/cluster/cellType/{cond}/CellLabels_F{fov_id[cond -1]}.png')
                    elif args.data_name == 'Xenium':
                        fov_img = Image.open(
                            f'Data/Xenium/Lung{cond}/dapi/{fov_id[cond -1]}.jpg')
                        cat_img = Image.open(
                            f'Experiment/Xenium/cluster/kmeans_2_clusters/{cond}/{fov_id[cond -1]}_msk.png')
                    bdy = bdry[cond - 1]
                    fov_img, cat_img = np.array(fov_img), np.array(cat_img)
                    Image.fromarray(fov_img[bdy[0]:bdy[1],
                                            bdy[0]:bdy[1]]).save(str(stat_pth / 'crop.png'))
                    run_fov_edit(args, stat_pth,
                                 fov_img, cat_img, bdy,
                                 dload, eigen, model,
                                 gene_idx=gene_idx)


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    setup_log(args)
    main(args)
