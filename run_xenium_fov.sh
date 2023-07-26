python -m analysis \
    --seed=7 \
    --data_name=Xenium \
    --analysis=GANI --task=fov_demo \
    --data_splt=slide_ID_numeric \
    --gene_num=392 \
    --n_eval=32 --n_iter=700000 \
    --data_path=Data/Xenium/GAN \
    --ckpt_path=Data/Xenium/GANI/style2_tabular_3_True_False_False_GAN_PARM_N_True_0.5_slide_ID_numeric_3_512/checkpoints \
    --save_path=Experiment_nm/Xenium/ --subtype

python -m analysis \
    --seed=7 \
    --data_name=Xenium \
    --analysis=GANI --task=fov_demo \
    --data_splt=slide_ID_numeric \
    --gene_num=392 \
    --n_eval=32 --n_iter=700000 \
    --data_path=Data/Xenium/GAN \
    --ckpt_path=Data/Xenium/GANI/style2_tabular_3_True_False_False_GAN_PARM_N_True_0.5_slide_ID_numeric_3_512/checkpoints \
    --save_path=Experiment_nm/Xenium/ --subtype --gene_name=MUC1,KRT7,RBM3,EPCAM