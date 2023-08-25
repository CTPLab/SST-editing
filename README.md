# In-silico-editing 
Official PyTorch implementation for the manuscript:

**In silico spatial transcriptomic editing at single-cell resolution**

Jiqing Wu and Viktor H. Koelzer.




## Demo 
### CosMx
$\triangledown$ Edit *HLA* and *B2M* (Normal -> Tumor)

https://github.com/CTPLab/In-silico-editing/assets/12932355/4efdfce1-55e5-496f-b0bc-3c9acfdb045a

$\triangledown$ Edit all genes (Normal -> Tumor)

https://github.com/CTPLab/In-silico-editing/assets/12932355/e7dc6bab-c04a-4c4b-84b3-25d298e2cb6a

$\triangledown$ Edit *HLA* and *B2M* (Tumor -> Normal)

https://github.com/CTPLab/In-silico-editing/assets/12932355/3eb34fa9-cdc0-4479-8e26-acbadbc7dc75

$\triangledown$ Edit all genes (Tumor -> Normal)

https://github.com/CTPLab/In-silico-editing/assets/12932355/e6e72a7f-c8c0-4525-8514-b95beceee4ac


### Xenium
$\triangledown$ Edit *MUC1*, *KRT7*, *RBM3*, and *EPCAM*  (Normal -> Tumor)

https://github.com/CTPLab/In-silico-editing/assets/12932355/4cb810f6-f41f-451f-9bdf-aa12d9468331

$\triangledown$ Edit all genes (Normal -> Tumor)

https://github.com/CTPLab/In-silico-editing/assets/12932355/646aca2e-9bd3-4ea4-bed2-0bd517520cc9

$\triangledown$ Edit *MUC1*, *KRT7*, *RBM3*, and *EPCAM* (Tumor -> Normal)

https://github.com/CTPLab/In-silico-editing/assets/12932355/ae973c82-4d89-4224-acea-456e7f215cb6

$\triangledown$ Edit all genes (Tumor -> Normal)

https://github.com/CTPLab/In-silico-editing/assets/12932355/60dadb00-0178-490f-b3da-c9f1dd976937

## Prerequisites
This implementation has been successfully tested under the following configurations:

- Ubuntu 20.04
- Nvidia driver 515.65
- CUDA 11.7
- Python 3.9
- PyTorch 2.0
- Miniconda 

Please use also [environment.yml](environment.yml) to create the conda environment for this repo.

## Preparation
First, we need the processed ST datasets:

1. CosMx: Unzip the downloaded [CosMx_crop](https://zenodo.org/record/8186465/files/cosmx_crop.zip?download=1)  to Data/ folder (create it if not exists) 

2. Xenium: Unzip the downloaded [Xenium_crop](https://zenodo.org/record/8186465/files/xenium_crop.zip?download=1) to Data/ folder


To reproduce the analysis results, we need some pre-computed weights and stats 

1. SCLIP pre-trained weights: Unzip the downloaded [SCLIP.zip](https://zenodo.org/record/8186465/files/SCLIP.zip?download=1) to the [SCLIP/](SCLIP) folder


2. Stats: Unzip the downloaded [stats.zip](https://zenodo.org/record/8186465/files/stats.zip?download=1) folder to this repo

3. CosMx:

    1. Unzip the downloaded [CosMx_GAN.zip](https://zenodo.org/record/8186465/files/CosMx_GAN.zip?download=1) weights to Data/CosMx/GAN folder

    2. Unzip the downloaded [CosMx_GANI.zip](https://zenodo.org/record/8186465/files/CosMx_GANI.zip?download=1) weights to Data/CosMx/GANI folder

4. Xenium: 

    1. Unzip the downloaded [Xenium_GAN.zip](https://zenodo.org/record/8186465/files/Xenium_GAN.zip?download=1) weights to Data/Xenium/GAN folder

    2. Unzip the downloaded [Xenium_GANI.zip](https://zenodo.org/record/8186465/files/Xenium_GANI.zip?download=1) weights to Data/Xenium/GANI folder
    

To reproduce the video demo results, we need the raw image data


1. CosMx:

    1. Download [F097.jpg](https://zenodo.org/record/8186465/files/CellComposite_F097.jpg?download=1) image to Data/CosMx/Liver1/CellComposite/ folder

    2. Download [F055.jpg](https://zenodo.org/record/8186465/files/CellComposite_F055.jpg?download=1) image to Data/CosMx/Liver2/CellComposite/ folder

    

2. Xenium: 

    1. Download [1.fig](https://zenodo.org/record/8186465/files/16000_20000_4000_8000_15800_20200_3800_8200.jpg?download=1) image to Data/Xenium/Lung1/dapi folder

    2. Download [2.fig](https://zenodo.org/record/8186465/files/12000_16000_32000_36000_11800_16200_31800_36200.jpg?download=1) image to Data/Xenium/Lung2/dapi folder

## Train the customized StyleGAN2 model 

Once the processed datasets are ready, we show the example script of training the model with two GPUs.
### CosMx
```
sh train_s2_cosmx.sh
```


### Xenium
```
sh train_s2_xenium.sh
```


## Train the customized StyleGAN2 Inversion model 

To train the GAN Inversion model, first download [moco](https://drive.google.com/file/d/1-1QfREW1Gz15sCU9w9pp9joAiQzTyNya/view?usp=sharing) to style3/pretrained_models folder

### CosMx
```
sh train_s2i_cosmx.sh
```


### Xenium
```
sh train_s2i_xenium.sh
```


## Run the analysis reported in the paper

### CosMx

Fig. 1 (b, c)
```
sh run_cosmx_plot.sh
```

Fig. 1 (d)
```
sh run_cosmx_metric.sh
```

Fig. 1 (e) (Results may vary due to random seed)
```
sh run_cosmx_demo.sh
```

Fig. 1 (g, h)
```
sh run_cosmx_fov.sh
```



### Xenium

Fig. 2 (a, b)
```
sh run_xenium_plot.sh
```

Fig. 2 (c)
```
sh run_xenium_metric.sh
```

Fig. 2 (d)  (Results may vary due to random seed)
```
sh run_xenium_demo.sh
```

Fig. 2 (f, g)
```
sh run_xenium_fov.sh
```

## Acknowledgment
This repository is built upon [Restyle-encoder](https://https://github.com/yuval-alaluf/restyle-encoder) and [StyleGAN3-editing](https://github.com/yuval-alaluf/stylegan3-editing.git) projects. We would like to thank all the authors contributing to those projects.
We would also like to thank all the authors contributing to the CosMx and Xenium datasets.

## License
The copyright license of this repository is specified with the LICENSE-In-silico-editing.

