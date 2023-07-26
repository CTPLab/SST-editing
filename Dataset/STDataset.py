import sys
import cv2
import torch
import sparse
import pickle
import random
import numpy as np
import pandas as pd
import spconv.pytorch as spconv
import torchvision.transforms.functional_tensor as F

from pathlib import Path
from PIL import Image, ImageFile

from Dataset.transform import transform_sp
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('.')


class STDataset(WILDSDataset):
    def __init__(self,
                 data,
                 gene_num,
                 gene_spa=False,
                 root_dir=Path('Data'),
                 transform=None,
                 split_scheme=None,
                 flter_subset=None,
                 debug=False,
                 seed=None,
                 # compatible to stylegan3
                 resolution=128,
                 num_channels=3,
                 max_size=None,
                 fov_demo=None,
                 sub_type=None):
        r"""
        The Dataset class of loading processed spatial transcriptomic data.
        After processing the raw data of the CosMx, Xenium, (later Visum) platforms,
        we have the matched and cleaned metadata_{img, cell}.csv, noting that
        metadata.csv, meatadata_img.csv, and metadata_cell.csv has the exact same row order
        -gene_expression one-dim vector
        -n-channel cellular images

        Args:
            data: Name to the dataset
            gene_num: Number of genes
            gene_spa: Whether take the sparse gene expression array as the input
            root_dir: Path to the root of data folder
            transform: Pytorch transform module
            split_scheme: Split dataset for calculating the quantification 
                          w.r.t. different class of the dataset, e.g., normal vs tumor,
            flter_subset: Filter the subset of the total data or subclass,
                          e.g., normal epithelial vs tumor epithelial.
            debug: Check the consistency between summed gene vector and 
                   gene table stored in the raw data, or if meatadata_img.csv does not
                   exist then create meatadata_img.csv
            seed: Random seed               
        """

        self._dataset_name = data
        self.gene_num = gene_num
        self.gene_spa = gene_spa
        self.img_dir = root_dir / f'{data}/GAN/crop'
        self.trans = transform
        self.debug = debug
        # This is for compatible to stylegan3 training
        if seed:
            random.seed(seed)
        self.subset = flter_subset
        self.resolution = resolution
        self.num_channels = num_channels

        # Prep metadata including cropped image paths
        df = pd.read_csv(str(self.img_dir / 'metadata.csv'), index_col=0)
        if self.subset is not None:
            df = df[self.subset]

        self._input_array = df.index.values
        self.ext = self._input_array[0].split('_')[-1]
        print(f'Path extension of {data}: {self.ext}')

        # Prep gene expression table derived from the center-cropped sparse gene array
        expr_pth = self.img_dir / 'metadata_img.csv'
        if self.debug and data == 'Xenium' and expr_pth.is_file():
            # *bar.csv record the numeric cell id stored in the zarr mask array
            expr_pth = self.img_dir / 'metadata_bar.csv'

        self.expr_img = self.prep_gene(expr_pth,
                                       True)

        # Prep gene expression table from the raw data corresponds to single-cell
        # which mostly used for debugging
        self.expr_cell = self.prep_gene(self.img_dir / 'metadata_cell.csv',
                                        False)

        # Prep subsetdata for downstream analysis
        self.prep_split(df, split_scheme)

    def prep_gene(self, gene_pth, load_name=True):
        r"""
        The function of loading the table (array) of gene expressions 
        and list of gene names 

        Args:
            gene_pth: Path to the csv file of gene expression table   
        """

        if not self.debug:
            assert gene_pth.is_file()

        if gene_pth.is_file():
            _expr = pd.read_csv(str(gene_pth),
                                index_col=0)
            # self.index is mainly for Xenium Lung cancer
            # as cellid is not number anymore
            # need to map cellid to the index of nucleus_boundaries
            if self.subset is not None:
                _expr = _expr[self.subset]
            if load_name:
                self.index = _expr.index.values.tolist()
            if self._dataset_name in ('CosMx', 'Xenium'):
                _expr = _expr.astype(np.int16)
            elif self._dataset_name == 'Visium':
                _expr = _expr.astype(np.float32)
            _expr = _expr.to_numpy()
        else:
            _expr = None
            print(f'{str(gene_pth)} does not exist')

        # list of gene names
        if load_name:
            with open(str(self.img_dir / 'transcripts.pickle'), 'rb') as fp:
                self.gene_name = pickle.load(fp)
            assert self.gene_num == len(self.gene_name)

        return _expr

    def prep_split(self, df, split_scheme):
        r"""
        The function of preparing the data split based on 
        self._split_array

        Args:
            df: Dataframe of metadata.csv
        """

        if split_scheme:
            t_nam, t_cnt = np.unique(df[split_scheme].values,
                                     return_counts=True)
            # if the counts have repetitive values, then it cannot be used
            # for stratify subset, thus add another _cond_dct
            self._cond_dct = dict(zip(t_nam, t_cnt))
            self._split_dict = dict(zip(t_nam, list(range(len(t_nam)))))
            self._split_array = df[split_scheme].map(self._split_dict).values
            self._y_array = torch.LongTensor(self._split_array)
            self._n_classes = len(t_nam)
        else:
            # add dummy metadata
            self._cond_dct = {None: len(self._input_array)}
            self._split_dict = {None: 0}
            self._split_array = np.zeros(len(self._input_array))
            self._y_array = torch.zeros(len(self._input_array))
            self._n_classes = 0

        # The coordinates stored in the self._metadata_fields
        # can be used for image/video creation
        self._y_size = 1
        if self.subset is not None:
            meta_list = []
            if self._dataset_name == 'CosMx':
                self._metadata_fields = ['y_FOV_px', 'x_FOV_px',]
            elif self._dataset_name == 'Xenium':
                self._metadata_fields = ['local_y', 'local_x',]
            for key in self._metadata_fields:
                meta_list.append(df[key].values.tolist())
            self._metadata_array = list(zip(*meta_list))
        else:
            self._metadata_array = list(zip(*[self._split_array.tolist()]))

    def get_input(self, idx):
        r"""
        The key function of loading paired image data and 
        gene expression data that are fed to GAN (Inversion) model
        or run debug function

        Args:
            idx: Row idx of the csv file of metadata
        """

        img_pth = self.img_dir / self._input_array[idx]
        gene_pth = str(img_pth).replace(self.ext, 'rna.npz')

        if not self.debug:
            img, gene_expr = self.run_input(img_pth, gene_pth, idx)
        else:
            img = torch.empty([0, 128, 128])
            if self.expr_img is not None:
                # This snippet used for collecting stats for
                # testing the gene expression consistency
                # between center-cropped gene array and metadata_cell.csv
                img, gene_expr = self.run_debug(
                    self.expr_img[idx], gene_pth, idx)
            else:
                # This snippet for creating metadata_img.csv
                gene_expr = sparse.load_npz(gene_pth)
                if self._dataset_name in ('CosMx', 'Xenium'):
                    gene_expr = gene_expr.sum((0, 1)).todense()
                    gene_expr = gene_expr.astype(np.int16)
        return img, gene_expr

    def run_input(self, img_pth, gene_pth, idx):
        r"""
        The key function of loading paired image data and 
        gene expression data that are fed to GAN (Inversion) model
        or run debug function

        Args:
            img_pth: Path to the image 
            gene_pth: Path to the sparse gene array, 
                      not useful when gene expression vector
                      from metadata_img.csv is loaded
            idx: Row idx of the csv file of metadata for loading 
                 the gene expression associated with the {idx} image
        """

        img = Image.open(img_pth).convert('RGB')
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous().float()
        if self._dataset_name == 'Xenium':
            img = img[0][None]
        elif self._dataset_name == 'CosMx':
            img = img[1:]
        if img.shape[1] != 128 or img.shape[2] != 128:
            img = F.resize(img, 128, 'bicubic', True).round().clamp(0, 255)

        if self.gene_spa:
            assert self._dataset_name in ('CosMx', 'Xenium')
            gene_expr = sparse.load_npz(gene_pth)
            # img = torch.from_numpy(gene_expr.todense().transpose((2, 0, 1)))
            if self.trans is not None:
                img, gene_expr = self.trans([img, gene_expr])

            # # naive init sparse tensor, incompatible to spconv
            # i = torch.LongTensor(np.array(gene_expr.coords))
            # v = torch.FloatTensor(np.array(gene_expr.data))
            # s = torch.Size(gene_expr.shape)
            # gene_expr = torch.sparse.FloatTensor(i, v, s)

            # init sparse tensor compatible to spconv
            i = torch.LongTensor(np.array(gene_expr.coords[:-1]))
            # create channel matrix indices
            c = gene_expr.coords[-1]
            r = list(range(len(c)))
            v = torch.zeros([len(c), self.gene_num]).float()
            # the data v is nothing but the matrix that
            # each row correspondes to the gene_num of readouts at certain spatial loc
            v[r, c] = torch.from_numpy(gene_expr.data.astype(np.float32))
            s = torch.Size(gene_expr.shape)
            gene_expr = torch.sparse_coo_tensor(i, v, s)

            # gcmp = torch.from_numpy(self.expr_img[idx]).contiguous().float()
            # gcmp1 = gene_expr.to_dense().sum((0, 1))
            # gcmp2 = gene_expr.coalesce().values().sum((0))
            # assert (gcmp == gcmp1).all() and (gcmp == gcmp2).all()
            # assert((img == gene_expr.to_dense().permute((2, 0, 1))).all())
        else:
            gene_expr = self.expr_img[idx]
            gene_expr = torch.from_numpy(gene_expr).contiguous().float()

            if self._n_classes > 0:
                # Append label for stylegan conditional training
                label = torch.nn.functional.one_hot(self._y_array[idx],
                                                    num_classes=self._n_classes)
                gene_expr = torch.cat([gene_expr, label])

            if self.trans is not None:
                img = self.trans(img)
        return img / 127.5 - 1, gene_expr

    def run_debug(self, gene_expr, gene_pth, idx):
        r"""
        The debug function for comparing the gene expression from metadata_img.csv
        and the one summed upon the sparse array, and further comparing gene expression
        from metadata_cell.csv (cell only gene expression) and the one summed upon the 
        sparse array masked with cell only region

        Args:
            gene_expr: The gene expression vector from metadata_image.csv
            gene_pth: Path to the sparse gene array, 
            idx: Row idx of the csv file of metadata for 
                 obtaining the cell id and loading the cell-only gene expression
        """

        out = sparse.load_npz(gene_pth).todense()
        if self._dataset_name in ('CosMx', 'Xenium'):
            assert (out.sum((0, 1)) == gene_expr).all()
            if self._dataset_name == 'CosMx':
                cell_pth = gene_pth.replace('rna.npz', 'cell.png')
                cell_np = cv2.imread(cell_pth, flags=cv2.IMREAD_UNCHANGED)
                # dir/Liver1/c_1_10_100_rna.npz, cid = 100
                cid = int(Path(gene_pth).name.split('_')[-2])
                out[cell_np != cid] = 0
            else:
                nucl_pth = gene_pth.replace('rna.npz', 'nucl.npz')
                nucl_coo = sparse.load_npz(nucl_pth).todense()
                cell_pth = gene_pth.replace('rna.npz', 'cell.npz')
                cell_coo = sparse.load_npz(cell_pth).todense()
                # dir/Rep1/*_*_1234_*_*_*_*_*_*_*_*_rna.npz, cid = 1234
                cid = int(Path(self.index[idx]).name.split('_')[2])
                out[(nucl_coo != cid) & (cell_coo != cid)] = 0
            out = np.abs(out.sum((0, 1)) -
                         self.expr_cell[idx])
        else:
            assert (out == gene_expr).all()
        return out, gene_expr


def proc_meta_img(data, num, bat=32):
    r"""
    The function of either debugging the gene expression 
    consistency or processing the metadata_img.csv

    Args:
        data: Name of the ST platform
        num: Number of genes 
                    not useful when gene expression vector
                    from metadata_img.csv is loaded
        bat: Batch size
    """

    ST = STDataset(data, num, debug=True)

    dload = get_eval_loader('standard',
                            ST, bat,
                            **{'drop_last': False,
                               'num_workers': 8,
                               'pin_memory': True})
    if ST.expr_img is not None:
        errn, errc = 0, 0
        for i, ((out, gene), _, _) in enumerate(dload):
            if data in ('CosMx', 'Xenium'):
                errn += out.sum()
                errc += (out.sum((-1)) != 0).sum()
            if (i + 1) % 1000 == 0:
                print(i + 1,
                      errn / (bat * (i + 1)),
                      errc / (bat * (i + 1)),)
        print(errn / len(ST), errc / len(ST))
    else:
        # Create the metadata_img for training
        gene_expr = torch.empty((0, num)).to(torch.int16)
        for i, ((img, gene), _, _) in enumerate(dload):
            gene_expr = torch.cat([gene_expr, gene.to(torch.int16)])
            if (i + 1) % 1000 == 0:
                print(i + 1)
        gene_expr = gene_expr.numpy()
        print(gene_expr.dtype, gene_expr.shape)
        csv_pth = f'Data/{data}/GAN/crop/metadata_img.csv'
        pd.DataFrame(gene_expr,
                     columns=ST.gene_name,
                     index=ST._input_array).to_csv(csv_pth)


def test_sparse_loader(data, num, trans=None, use_spconv=False, bat=16):
    if use_spconv:
        toy = spconv.SparseSequential(
            spconv.SparseConv2d(num, 32, 3, padding=1),
            spconv.ToDense()).cuda()

    ST = STDataset(data, num, gene_spa=True, transform=trans)
    print(data, len(ST))
    dload = get_train_loader('standard', ST, bat)
    for i, ((img, gene_spa), _, _) in enumerate(dload):
        # gsum = gene_spa.to_dense().sum((1, 2))
        # assert (gsum == gene_tab).all()

        if use_spconv:
            # gene_spa = gene_spa.coalesce().cuda()
            # gene_inp = spconv.SparseConvTensor.from_dense(gene_spa)
            # gene_feat = toy(gene_inp)
            gene_spa1 = gene_spa.to_dense().cuda()
            gene_inp1 = spconv.SparseConvTensor.from_dense(
                gene_spa1.to_sparse(3))
            gene_feat1 = toy(gene_inp1)
            # assert torch.allclose(gene_feat, gene_feat1)

        if i % 1000 == 0:
            print(i, img.shape, gene_spa.shape)


if __name__ == '__main__':
    # This part used for generating (or testing) metadata_img.csv for three datasets
    for (d, n) in (('CosMx', 1000),):
        # when d == visium, run debug test instead of creating metadata_img.csv
        proc_meta_img(d, n)

    # # If test on 256 x 256, it can be slow
    # for (d, n) in (('CosMx', 1000), ('Xenium', 280)):
    #     for t in (transform_sp, None):
    #         test_sparse_loader(d, n, t, True)
