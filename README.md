# Intro
The official code for the following paper :

**Boudary-aware Image Inpainting with Multiple Auxiliary Cues**, NTIRE2022

Download paper [here](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yamashita_Boundary-Aware_Image_Inpainting_With_Multiple_Auxiliary_Cues_CVPRW_2022_paper.pdf)

# Prerequisites
* Python3
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

# Installation  

1.Clone the repository

```
git clone https://github.com/rain58/Boudary-aware-Image-Inpainting.git  
cd Boundary-aware-Image-Inpainting
```

2.To create Python environment:
```
pip install -r requirements.txt
```

# Datasets
## Images
We use [Places2](http://places2.csail.mit.edu/) and [Paris Street-View](https://github.com/pathak22/context-encoder).Please download datasets from their official website.
Arter downloading, make flist files:
```
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```
## Masks
We use mask datasets provided by [Liu et al.](https://arxiv.org/abs/1804.07723).
You can download datasets from [here](http://masc.cs.gmu.edu/wiki/partialconv)   
Arter downloading, make flist files:
```
python ./scripts/flist.py --path path_to_mask_train_set --output ./datasets/mask_train.flist
```

# Training


#Testing
Download the pre-trained models from the following links and copy them under `./checkpoints`
our pretrained model is [here](https://drive.google.com/drive/folders/1GOGqqkOKjS3N2aXRe_7tynJ58gDfJIme?usp=sharing)

