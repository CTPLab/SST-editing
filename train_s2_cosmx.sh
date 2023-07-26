python -m torch.distributed.launch --nproc_per_node=2 --master_port=4567 train_style2.py \
    Data/CosMx/GAN/crop \
    --data=CosMx \
    --gene=1000 \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --kernel_size=3 \
    --gene_use \
    --split_scheme=slide_ID_numeric \
    --img_chn=2 --latent=512 --mixing=0.5 \
    --check_save=Data/CosMx/GAN/N/