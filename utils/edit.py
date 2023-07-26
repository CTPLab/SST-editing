import copy
import cv2
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import torchvision.utils as tv_utils

from PIL import Image
from utils.common import run_GAN, run_GANI, prep_input, prep_outim, add_bbx


def run_one_demo(analysis, decoder,
                 model, img, gene, noise=None, shift=None):
    r"""
    Run the GAN (Inversion) model once to generate one video frame

    Args:
        analysis: The model used for analysis: GAN or GAN Inversion
        decoder: StyleGAN2 generator
        model: GAN or GAN Inversion
        img: morphological image
        gene: one-dim gene expression vector
        noise: Gaussian noise fed to the mapping of StyleGAN2
        shift: The morphological transition coefficient,
               fully transformed to the reference cell morphology 
               if shift == 1. Only for GAN Inversion
    """

    with torch.inference_mode():
        if analysis == 'GAN':
            out = run_GAN(decoder, model, gene, noise,
                          randomize_noise=False)
        elif 'GANI' in analysis:
            out = run_GANI(decoder, model, gene, img,
                           randomize_noise=False,
                           shift=shift)
    return out.clamp(-1, 1)


def run_demo_edit(args, path,
                  dload, eigen, model,
                  gene_idx, step=0.01, target_only=False):
    r"""
    Create cell image gallary for GAN (Inversion) model.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        dload: Dataloader
        eigen: List of eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN (Inversion) model
        gene_idx: Indices of targeted gene expressions
        step: the step length of morph transition for each video frame
        target_only: set non-targeted genes to 0 if True (not very useful)
    """

    noise_fix = torch.randn((args.n_eval, 512))
    for i,  ((img, gene), _, _) in enumerate(dload):
        wrt = imageio.get_writer(str(path / f'{str(i).zfill(3)}.mp4'),
                                 fps=24)
        with torch.inference_mode():
            img, gene = prep_input(img, gene, args.gene_num,
                                   is_GAN=args.analysis == 'GAN')
            out = run_one_demo(args.analysis, args.decoder,
                               model, img, gene, noise_fix,
                               shift=0)
            ovid = torch.cat(out.unbind(0), -1)
            ivid = torch.cat(img.unbind(0), -1)
            gene_all = edit_gene(gene, eigen[0], eigen[1])
            gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)

            for shf in np.arange(0, 1 + step, step):
                shf_all = (1 - shf) * gene + shf * gene_all
                shf_sub = (1 - shf) * gene + shf * gene_sub
                if target_only:
                    msk = torch.zeros(shf_sub.shape[-1]).to(gene)
                    msk[gene_idx] = 1
                    shf_sub = msk * shf_sub
                out_all = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_all, noise_fix,
                                       shift=shf)
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)

                if shf == 1:
                    out_img = (torch.cat([img, out, out_sub, out_all]) + 1) / 2
                    if out_img.shape[1] == 2:
                        out_img = torch.cat((torch.zeros_like(out_img[:, 0])[:, None],
                                             out_img), dim=1)
                    tv_utils.save_image(out_img,
                                        str(path / f'{str(i).zfill(3)}.png'),
                                        nrow=args.n_eval,
                                        padding=2)

                output = torch.cat([ivid,
                                    ovid,
                                    torch.cat(out_sub.unbind(0), -1),
                                    torch.cat(out_all.unbind(0), -1)], 1)
                if output.shape[0] == 2:
                    output = torch.cat((-torch.ones_like(output[0])[None],
                                        output))
                output = output.transpose(0, 2).transpose(0, 1)
                output = prep_outim(output).cpu().numpy()
                wrt.append_data(output.astype('uint8'))
            wrt.close()
            return


def run_fov_edit(args, path,
                 fov, cat, bdy,
                 dload, eigen, model,
                 gene_idx, step=0.01, target_only=False):
    r"""
    Create ROI video demo for GAN Inversion model, which
    bdy parameters are customized for the selected ROI image.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        fov: Morpholocial image of region of interest
        cat: Cluster annotation image of ROI
        bdy: Boundary of the image to be cropped for demo
        dload: Dataloader
        eigen: list of eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN Inversion model
        gene_idx: Indices of targeted gene expressions
        step: the step length of morph transition for each video frame
        target_only: set non-targeted genes to 0 if True (not very useful)
    """

    if args.data_name == 'CosMx':
        sz = 160
    elif args.data_name == 'Xenium':
        sz = 96

    noise_fix = torch.randn((args.n_eval, 512))
    wrt = imageio.get_writer(str(path / 'fov_demo.mp4'),
                             fps=24)
    for shf in np.arange(0, 1 + step, step):
        print(shf)
        fov_edit = copy.deepcopy(fov).astype(np.float32)
        msk_edit = np.zeros_like(fov)
        if shf == 0:
            fov_bbx = copy.deepcopy(fov)
        for i,  ((img, gene), _, meta) in enumerate(dload):
            with torch.inference_mode():
                img, gene = prep_input(img, gene, args.gene_num,
                                       is_GAN=args.analysis == 'GAN')
                gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)
                shf_sub = (1 - shf) * gene + shf * gene_sub
                if target_only:
                    msk = torch.zeros(shf_sub.shape[-1]).to(gene)
                    msk[gene_idx] = 1
                    shf_sub = msk * shf_sub
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)
                out_sub = F.interpolate(out_sub, sz,
                                        mode='bicubic', antialias=True)
                out_sub = prep_outim(out_sub)
                if out_sub.shape[1] == 2:
                    out_sub = torch.cat((torch.zeros_like(out_sub[:, 0])[:, None],
                                         out_sub), dim=1)
                out_sub = out_sub.transpose(1, 3).transpose(1, 2)
                if len(out_sub.shape) == 3:
                    out_sub = out_sub.squeeze(-1)
                for b in range(img.shape[0]):
                    if (b % 8 != i % 8 and args.data_name == 'Xenium') or \
                       (b % 8 != 1 and args.data_name == 'CosMx'):
                        continue
                    r = slice(meta[0][b] - sz // 2, meta[0][b] + sz // 2)
                    c = slice(meta[1][b] - sz // 2, meta[1][b] + sz // 2)
                    if (msk_edit[r, c] == 0).all():
                        msk_edit[r, c] = 1
                        img_crop = out_sub[b].cpu().numpy()
                        if args.data_name == 'Xenium':
                            img_crop = add_bbx(
                                np.repeat(img_crop, 3, -1))[:, :, 0]
                        else:
                            img_crop = add_bbx(img_crop)
                        fov_edit[r, c] = img_crop

                        if shf == 0:
                            cat[r, c] = add_bbx(cat[r, c])
                            if args.data_name == 'Xenium':
                                fov_bbx[r, c] = add_bbx(
                                    np.repeat(fov_bbx[r, c][:, :, None], 3, -1))[:, :, 0]
                            else:
                                fov_bbx[r, c] = add_bbx(fov_bbx[r, c])
        fov_edit = fov_edit[bdy[0]:bdy[1], bdy[0]:bdy[1]].astype('uint8')
        if args.data_name == 'Xenium':
            # This is configured for the selected lung ROI image
            fov_edit = fov_edit[-1024:, -1024:]
        wrt.append_data(fov_edit)
        if shf == 0 or shf == 1:
            im_nm = 'rec' if shf == 0 else 'edit'
            Image.fromarray(fov_edit).save(str(path / f'{im_nm}.png'))
            if shf == 0:
                Image.fromarray(fov_bbx[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'crop_bbx.png'))
                Image.fromarray(cat[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'cat_bbx.png'))
    wrt.close()
    return


def run_fov_edit1(args, path,
                  fov, cat, bdy,
                  dload, eigen, model,
                  gene_idx, step=0.5, target_only=False):
    if args.data_name == 'CosMx':
        sz = 160
    elif args.data_name == 'Xenium':
        sz = 96

    stit = Image.open('utils/msk.png').resize((sz, sz))
    stit = np.array(stit).astype(np.float32) / 255
    if args.data_name == 'CosMx':
        stit = stit[:, :, None]

    noise_fix = torch.randn((args.n_eval, 512))
    wrt = imageio.get_writer(str(path / 'fov_demo.mp4'),
                             fps=24)
    for shf in np.arange(0, 1 + step, step):
        print(shf)
        # fov_edit = copy.deepcopy(fov).astype(np.float32)
        fov_edit = np.zeros_like(fov).astype(np.float32)
        msk_edit = np.zeros_like(fov).astype(np.float32)
        if shf == 0:
            fov_bbx = copy.deepcopy(fov)
        for i,  ((img, gene), _, meta) in enumerate(dload):
            with torch.inference_mode():
                img, gene = prep_input(img, gene, args.gene_num,
                                       is_GAN=args.analysis == 'GAN')
                gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)
                shf_sub = (1 - shf) * gene + shf * gene_sub
                if target_only:
                    msk = torch.zeros(shf_sub.shape[-1]).to(gene)
                    msk[gene_idx] = 1
                    shf_sub = msk * shf_sub
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)
                out_sub = F.interpolate(
                    out_sub, sz, mode='bicubic', antialias=True)
                out_sub = prep_outim(out_sub)
                if out_sub.shape[1] == 2:
                    out_sub = torch.cat((torch.zeros_like(out_sub[:, 0])[:, None],
                                         out_sub), dim=1)
                out_sub = out_sub.transpose(1, 3).transpose(1, 2)
                if len(out_sub.shape) == 3:
                    out_sub = out_sub.squeeze(-1)
                for b in range(img.shape[0]):
                    # if b % 8 != i % 8:
                    # # if b % 8 != 1:
                    #     continue
                    r = slice(meta[0][b] - sz // 2, meta[0][b] + sz // 2)
                    c = slice(meta[1][b] - sz // 2, meta[1][b] + sz // 2)
                    out_stit = out_sub[b].squeeze().cpu().numpy() * stit
                    fov_edit[r, c] += out_stit
                    msk_edit[r, c] += stit
                    # if (msk_edit[r, c] == 0).all():
                    #     msk_edit[r, c] = 1
                    #     img_crop = out_sub[b].cpu().numpy()
                    #     if args.data_name == 'Xenium':
                    #         img_crop = add_bbx(np.repeat(img_crop, 3, -1))[:,:,0]
                    #     else:
                    #         img_crop = add_bbx(img_crop)
                    #     fov_edit[r, c] = img_crop

                    #     if shf == 0:
                    #         cat[r, c] = add_bbx(cat[r, c])
                    #         if args.data_name == 'Xenium':
                    #             fov_bbx[r, c] = add_bbx(np.repeat(fov_bbx[r, c][:,:,None], 3, -1))[:,:,0]
                    #         else:
                    #             fov_bbx[r, c] = add_bbx(fov_bbx[r, c])
        fov_edit[msk_edit != 0] /= msk_edit[msk_edit != 0].astype(np.float32)
        msk_edit[msk_edit != 0] = 1
        # fov_edit[msk_edit == 0] = fov[msk_edit == 0]
        fov_edit = np.clip(fov_edit, 0, 255)
        fov_edit = fov_edit[bdy[0]:bdy[1], bdy[0]:bdy[1]].astype('uint8')
        fov_edit = cv2.inpaint(fov_edit,
                               ((1 - msk_edit[bdy[0]:bdy[1], bdy[0]
                                :bdy[1], 0]) * 255).astype('uint8'),
                               3, cv2.INPAINT_NS)
        wrt.append_data(fov_edit)
        if shf == 0 or shf == 1:
            im_nm = 'rec' if shf == 0 else 'edit'
            Image.fromarray(fov_edit).save(str(path / f'{im_nm}.png'))
            if shf == 0:
                Image.fromarray(fov_bbx[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'crop_bbx.png'))
                Image.fromarray(cat[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'cat_bbx.png'))
    wrt.close()
    return


def edit_gene(x, eigx, eigr,
              wei=None, topk=1, idx=None):
    r"""
    The core function of in silico editing by matching 
    the SCM of gene expressions of a give cell population to another

    Args:
        x: The collection of gene expressions of 
           a given population
        eigx: Eigenvector and eigenvalue of x
        eigr: Eigenvector and eigenvalue of gene expressions
              from another cell population
        wei: Scaled weights added to the 
             transformation (not very useful)
        topk: The top k-th eigenvalues to be transformed, 
              topk = 1 by default
        idx: Indices of gene expressions to be transformed,
             transform all genes if ids=None 
    """

    if eigx is None:
        return x
    else:
        x = x.detach().clone()
        vecx, valx = eigx
        vecr, valr = eigr
        xedt = x @ vecx.float().to(x)
        if wei is None:
            wei = torch.ones_like(valx).float()
            wei[:topk] = (valr[:topk] / valx[:topk]).sqrt().float()
        xedt = wei[None].to(x) * xedt
        xedt = xedt @ vecr.T.float().to(x)
        if idx is not None:
            x[:, idx] = xedt[:, idx].clone()
            xedt = x
        xedt[xedt < 0] = 0
        return xedt
