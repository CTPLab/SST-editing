python -m analysis \
    --seed=0 \
    --analysis=RAW --task=gene_eig \
    --gene_num=392 --gene_use \
    --data_splt=slide_ID_numeric \
    --data_name=Xenium \
    --data_path=Data/Xenium/GAN \
    --save_path=Experiment/Xenium/

python -m analysis \
    --seed=0 \
    --analysis=RAW --task=gene_eig \
    --gene_num=392 --gene_use \
    --data_splt=slide_ID_numeric \
    --data_name=Xenium \
    --data_path=Data/Xenium/GAN \
    --save_path=Experiment/Xenium/ --subtype