python -m analysis \
    --seed=5 \
    --data_name=CosMx \
    --analysis=GANI --task=fov_demo \
    --data_splt=slide_ID_numeric \
    --gene_num=1000 \
    --n_eval=32 --n_iter=700000 \
    --data_path=Data/CosMx/GAN \
    --ckpt_path=Data/CosMx/GANI/style2_tabular_3_True_False_False_GAN_PARM_N_True_0.5_slide_ID_numeric_3_512/checkpoints \
    --save_path=Experiment_nm/CosMx/ --subtype

python -m analysis \
    --seed=5 \
    --data_name=CosMx \
    --analysis=GANI --task=fov_demo \
    --data_splt=slide_ID_numeric \
    --gene_num=1000 \
    --n_eval=32 --n_iter=700000 \
    --data_path=Data/CosMx/GAN \
    --ckpt_path=Data/CosMx/GANI/style2_tabular_3_True_False_False_GAN_PARM_N_True_0.5_slide_ID_numeric_3_512/checkpoints \
    --save_path=Experiment_nm/CosMx/ --subtype --gene_name=HLA-A,B2M