import json
import pprint
import pyrallis
import dataclasses
import spconv.pytorch as spconv

from Dataset.STDataset import STDataset
from Dataset.transform import transform_sp, transform

from style3.inversion.training.coach_restyle_psp import Coach
from style3.inversion.options.train_options import TrainOptions


@pyrallis.wrap()
def main(opts: TrainOptions):
    exp_nm = f'{opts.decoder}_{opts.gene_type}_{opts.kernel_size}_{opts.gene_use}_{opts.train_decoder}_{opts.use_cfgr}'
    if opts.stylegan_weights is not None:
        i = -3 if opts.decoder == 'style2' else -2
        sty_parm = str(opts.stylegan_weights).split('/')[i]
        exp_nm += f'_GAN_PARM_{sty_parm}'
    opts.exp_dir = opts.exp_dir / exp_nm
    opts.exp_dir.mkdir(exist_ok=True, parents=True)

    opts_dict = dataclasses.asdict(opts)
    pprint.pprint(opts_dict)
    with open(opts.exp_dir / 'opt.json', 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True, default=str)

    gene_spa = True if opts.gene_type == 'spatial' else False
    trans = transform_sp if opts.gene_type == 'spatial' else transform
    data = STDataset(opts.dataset_type, opts.gene_num, gene_spa,
                     split_scheme=opts.data_splt,
                     transform=trans)

    coach = Coach(opts, data)
    coach.train()


if __name__ == '__main__':
    main()
