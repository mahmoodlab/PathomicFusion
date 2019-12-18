# Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis

©[Mahmood Lab](www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for use for non-commercial academic purposes. 

If you use this code, please cite:
```
@inproceedings{chen2019pathomicfusion,
  title     = {Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis},
  author    = {Richard J. Chen, Ming Y. Lu, Jingwen Wang, Drew F. K. Williamson, Scott J. Rodig, Neal I. Lindeman, Faisal Mahmood},
  booktitle = {arXiv},
  year = {2019}
}
```
 
## Setup

### 1. Prerequisites
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Tis on local workstations, and Nvidia V100s using Google Cloud)
- CUDA + cuDNN (Tested on CUDA 10.1 and cuDNN 7.5. CPU mode and CUDA without CuDNN may work with minimal modification, but untested.)
- torch>=1.1.0
- torch_geometric=1.3.0

### 2. Code Base Structure
Code base organization is borrowed and inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), with most of the modularity for "running experiments on different datasets" removed for simplicity.
- **train_cv.py**: Cross-validation script for training unimodal and multimodal networks. This script will save evaluation metrics and predictions on the train + test split for each epoch on every split in **checkpoints**.
- **test_cv.py**: Script for testing unimodal and unimodal networks on only the test split.
- **train_test.py**: Contains the definitions for "train" and "test". 
- **networks.py**: Contains PyTorch model definitions for all unimodal and multimodal network.
- **fusion.py**: Contains PyTorch model definitions for fusion.
- **data_loaders.py**: Contains the PyTorch DatasetLoader definition for loading multimodal data.
- **options.py**: Contains all the options for the argparser.
- **make_splits.py**: Script for generating a pickle file that saves + aligns the path for multimodal data for cross-validation.
- **run_cox_baselines.py**: Script for running Cox baselines.
- **utils.py**: Contains definitions for collating, survival loss functions, data preprocessing, evaluation, figure plotting, etc...

The directory structure for your multimodal dataset should look similar to the following:
```bash
./
├── data
      └── PROJECT
            ├── INPUT A (e.g. Image)
                ├── image_001.png
                ├── image_002.png
                ├── ...
            ├── INPUT B (e.g. Graph)
                ├── image_001.pkl
                ├── image_002.pkl
                ├── ...
            └── INPUT C (e.g. Genomic)
                └── genomic_data.csv
└── checkpoints
        └── PROJECT
            ├── TASK X (e.g. Survival Analysis)
                ├── path
                    ├── ...
                ├── ...
            └── TASK Y (e.g. Grade Classification)
                ├── path
                    ├── ...
                ├── ...
```

Depending on which modalities you are interested in combining, you must: (1) write your own function for aligning multimodal data in **make_splits.py**, (2) create your DatasetLoader in **data_loaders.py**, (3) modify the **options.py** for your data and task. Models will be saved to the **checkpoints** directory, with each model for each task saved in its own directory. At the moment, the only supervised learning tasks implemented are survival outcome prediction and grade classification.

### 3. Training and Evaluation
Here are example commands for training unimodal + multimodal networks.

#### Survival Model for Input A
Example shown below for training a survival model for mode A and saving the model checkpoints + predictions at the end of each split. In this example, we would create a folder called "CNN_A" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "A" is defined as a mode in **dataset_loaders.py** for handling modality-specific data-preprocessing steps (random crop + flip + jittering for images), and that there is a network defined for input A in **networks.py**. "surv" is already defined as a task for training networks for survival analysis in **options.py, networks.py, train_test.py, train_cv.py**.

```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode A --model_name CNN_A --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```
To obtain test predictions on only the test splits in your cross-validation, you can replace "train_cv" with "test_cv".
```
python test_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode input_A --model input_A_CNN --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```

#### Grade Classification Model for Input A + B
Example shown below for training a grade classification model for fusing modes A and B. Similar to the previous example, we would create a folder called "Fusion_AB" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "AB" is defined as a mode in **dataset_loaders.py** for handling multiple inputs A and B at the same time. "grad" is already defined as a task for training networks for grade classification in **options.py, networks.py, train_test.py, train_cv.py**.
```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task grad --mode AB --model_name Fusion_AB --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```

#### 4. Reproducibility
To reporduce the results in our paper and for exact data preprocessing, implementation, and experimental details please follow the instructions here: [./data/TCGA_GBMLGG/](https://github.com/Richarizardd/pathomic_fusion/tree/master/data/TCGA_GBMLGG/) 

## Issues
- Please report all issues on the public forum.

## License
This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

## Disclosures 
- A provisional patent on this work has been filed by the Brigham and Women's Hospital.

## Acknowledgments
- This code is inspired by [SALMON](https://github.com/huangzhii/SALMON), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and [SCNN](https://github.com/CancerDataScience/SCNN).
* Subsidized computing resources were provided by Nvidia and Google Cloud.

## Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{chen2019pathomicfusion,
  title     = {Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis},
  author    = {Richard J. Chen, Ming Y. Lu, Jingwen Wang, Drew F. K. Williamson, Scott J. Rodig, Neal I. Lindeman, Faisal Mahmood},
  booktitle = {arXiv},
  year = {2019}
}
```
