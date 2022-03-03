# Reproducibility of "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis"

## Setup
### 1. Processed Dataset
Processed data can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf?usp=sharing).

The data directory structure for TCGA-GBM + TCGA-LGG validation is listed below. 
- **all_datasets.csv**: Contains survival time, censor status, IDH mutation status, and CNV data for 769 TCGA IDs.
- **grade_data.csv**: Contains age, gender, histologic grade and subtype data for 769 TCGA IDs.
- **mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt**: Contains mRNAseq data for the TCGA-GBM project (obtained from the top differentially expressed genes from [cBioPortal](https://www.cbioportal.org/)).
- **mRNA_Expression_Zscores_RSEM.txt**: Contains mRNAseq data for the TCGA-LGG project (obtained from the top differentially expressed genes from [cBioPortal](https://www.cbioportal.org/)).
- **pnas_splits.csv**: Splits from [Mobadersany et al.](https://github.com/CancerDataScience/SCNN) used for 15-fold cross-validation.
- **all_st**: 1505 1024 X 1024 histology ROIs for the 769 TCGA IDs (Stain Normalized) used for training Histology CNN
- **all_st_cpc**: Graph features for the 1505 histology ROIs used for training Histology GCN
- **all_st_patches_512**: 13545 512 X 512 patches (9 overlapping (stride = 256) patches extracted per image in all_st) used for testing Histology CNN, and training + testing Pathomic Fusion. Instead of random cropping, **all_st_patches_512** can be interpretted as fixed crops per image.
- **all_st_patches_512_cpc**: Graph features for the 13545 histology ROIs used for training Histology GCN. Since we did not need to use a patch-based strategy for training the GCN, these .pt files are .pt files duplicated from **all_st_cpc** to align the graph and image input before loading it in the PyTorch Dataset Loader.
- **splits**: Pickle files containing the data splits for 15-fold cross-validation. Depending on the task (grade vs. survival) or model being trained (CNN, GCN, SNN, Pathomic Fusion), missing data was excluded. In the pickle filename,  the string "all_st" vs. "all_st_patches_512" indicates that the genomic data was aligned with the 1024 X 1024 images in **all_st/all_st_cpc** or 512 X 512 images in **all_st_patches_512 / all_st_patches_512_cpc**. The ending string with pattern "INT_INT_INT_STR" indicates: 0/1 for if we should ignore patients with missing molecular subtype, 0/1 for if we should ignore patients with missing histology subtype, 0/1 for we should ignore patients with missing molecular subtype, 0/1 for if we should use extracted VGG19 embeddings from **all_st_patches_512** for Pathomic Fusion, and "rnaseq" for if we should use RNAseq. Additional details can be found in **make_splits.py**.
  
```bash
./
└── data
      └── TCGA_GBMLGG
            ├── all_datasets.csv
            ├── grade_data.csv
            ├── mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt
            ├── mRNA_Expression_Zscores_RSEM.txt
            ├── pnas_splits.csv
            ├── gbmlgg
                  ├── all_st
                        ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1.png
                        ├── TCGA-02-0001-01Z-00-DX2.b521a862-280c-4251-ab54-5636f20605d0_1.png
                        ├── ...
                  ├── all_st_cpc
                        └── pt
                            ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1.pt 
                            ├── TCGA-02-0001-01Z-00-DX2.b521a862-280c-4251-ab54-5636f20605d0_1.pt
                            ├── ...
                  ├── all_st_patches_512
                        ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1_0_0.png
                        ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1_0_256.png
                        ├── ...
                  ├── all_st_patches_512_cpc
                        └── pt
                            ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1_0_0.pt 
                            ├── TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1_0_256.pt 
                            ├── ...
            └── splits
                  ├── gbmlgg15cv_all_st_0_0_0.pkl
                  ├── gbmlgg15cv_all_st_0_1_0.pkl
                  ├── ...
      ├──  Other (Paired) Datasets :) 
```

### 2. Pretrained Models
All pretrained models and predictions can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf?usp=sharing), and are organized as follows below.
```bash
./
└── checkpoints
    ├── surv_15
        ├── path
            ├── path_1.pt
            ├── path_1_pred_train.pkl
            ├── path_1_pred_test.pkl
            ├── ...
        ├── ...
    └── grad_15
        ├── path
            ├── ...
        ├── ...
```
where "surv_15" and "grad_15" refers to the 15-fold cross-validation on Pathomic Fusion for survival outcome prediction and grade classification respectively.

### Training
Commands for training each model:

##### Histology CNN
```
python train_cv.py  --exp_name surv_15_rnaseq --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --gpu_ids 0
python test_cv.py  --exp_name surv_15_rnaseq --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --gpu_ids 0 --use_vgg_features 1
python train_cv.py  --exp_name grad_15 --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0
python test_cv.py  --exp_name grad_15 --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 1
```

##### Histology GCN
```
python train_cv.py  --exp_name surv_15_rnaseq --task surv --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 -use_vgg_features 1 --gpu_ids 0
python train_cv.py  --exp_name grad_15 --task grad --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 -use_vgg_features 1 --act LSM --label_dim 3 --gpu_ids 0
```

##### Genomic SNN
```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
python train_cv.py --exp_name grad_15 --task grad --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --act LSM --label_dim 3 --gpu_ids 0
```

##### Pathomic Fusion (CNN+SNN)
```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
python train_cv.py --exp_name grad_15 --task grad --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --path_gate 0 --omic_scale 2 --act LSM --label_dim 3
```

##### Pathomic Fusion (GCN+SNN)
```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
python train_cv.py --exp_name grad_15 --task grad --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --grph_gate 0 --omic_scale 2 --act LSM --label_dim 3
```

##### Pathomic Fusion (CNN+GCN+SNN)
```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
python train_cv.py --exp_name grad_15 --task grad --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_B --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --path_gate 0 --act LSM --label_dim 3
```


### Working with the Raw TCGA Data
 Raw histology region-of-interests for the TCGA-GBM and TCGA-LGG projects can be downloaded from [Mobadersany et al.](https://github.com/CancerDataScience/SCNN). For stain normalization, we used a python implementation of Sparse Stain Normalization from [Vahdane et al.](https://github.com/abhishekvahadane/CodeRelease_ColorNormalization) implemented in [StainTools](https://github.com/Peter554/StainTools). 
 
### Issues
- Please open new threads or report issues to richardchen@g.harvard.edu.

## License
This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
- This code is inspired by [SALMON](https://github.com/huangzhii/SALMON), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and [SCNN](https://github.com/CancerDataScience/SCNN).
* Subsidized computing resources were provided by Nvidia and Google Cloud.

## Reference
If you find our work useful in your research, please consider citing our paper at:
```
@article{chen2020pathomic,
  title={Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis},
  author={Chen, Richard J and Lu, Ming Y and Wang, Jingwen and Williamson, Drew FK and Rodig, Scott J and Lindeman, Neal I and Mahmood, Faisal},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}
```
© Mahmood Lab - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
