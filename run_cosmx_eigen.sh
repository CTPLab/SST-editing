python -m analysis \
    --seed=0 \
    --analysis=RAW --task=gene_eig \
    --gene_num=1000 --gene_use \
    --data_splt=slide_ID_numeric \
    --data_name=CosMx \
    --data_path=Data/CosMx/GAN \
    --save_path=Experiment/CosMx/

python -m analysis \
    --seed=0 \
    --analysis=RAW --task=gene_eig \
    --gene_num=1000 --gene_use \
    --data_splt=slide_ID_numeric \
    --data_name=CosMx \
    --data_path=Data/CosMx/GAN \
    --save_path=Experiment/CosMx/ --subtype