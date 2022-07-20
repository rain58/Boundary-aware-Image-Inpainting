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
## RGB Images
We use [Places2](http://places2.csail.mit.edu/) and [Paris Street-View](https://github.com/pathak22/context-encoder).Please download datasets from their official website.
Arter downloading, make flist files:
```
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```
## Depths
Estimate depth image from RGB Images datasets by using [Dense Depth](https://arxiv.org/abs/1812.11941).
The procedure is as follows.
1. Fine-tune the pre-trained Dense Depth model by using [DIODE dataset](https://arxiv.org/abs/1908.00463). We use only outdoor images of DIODE dataset. 
2. Estimate the depth image from RGB Images.

## Masks
We use mask datasets provided by [Liu et al.](https://arxiv.org/abs/1804.07723).
You can download datasets from [here](http://masc.cs.gmu.edu/wiki/partialconv)   
Arter downloading, make flist files:
```
python ./scripts/flist.py --path path_to_mask_train_set --output ./datasets/mask_train.flist
```

# Getting Started
## Training
To train the model, create a `config.yaml` file similar to the example config file and copy it under your checkpoints directory.  

Our model is trained in three stages:
1. training edge model or download the edge model from [EdgeConnect](https://github.com/knazeri/edge-connect)
2. training depth model
3. training the inpaint model  
To train the model, change the "model" option number in `train.sh`.
In details, Edge:1, Depth:5, Inpaint:6. Then, run `sh train.sh`

## Testing
Download the pre-trained models from the following links and copy them under `./checkpoints`
our pretrained model : [Paris Street-View](https://drive.google.com/drive/folders/1GOGqqkOKjS3N2aXRe_7tynJ58gDfJIme?usp=sharing), [Places]()

