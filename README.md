# SMART-Net - Official Pytorch Implementation

It's scheduled to be uploaded soon. We are modifying the code for easier use.
We proposed a supervised multi-task aiding representation transfer learning network called <b>SMART-Net</b>.


## Highlights
+ Develop a robust feature extractor of brain hemorrhage in head & neck NCCT through three kinds of
multi-task representation learning.
+ Propose the consistency loss to alleviate the disparity of two pretext tasks' heads, resulting in
improved transferability and representation.
Connect the feature extractor with the target-specific 3D operator via transfer learning to expand
volume-level tasks.
+ Explore relationships of the proposed multi-pretext task combinations and perform ablation studies on
optimal 3D operators for volume-level ICH tasks.
+ Validate the model on multiple datasets with previous methods and ablation studies for the robustness
and practicality of our method.
Highlights


<p align="center"><img width="100%" src="figures/graphical_abstract.png" /></p>
<!-- <p align="center"><img width="85%" src="figures/framework.png" /></p> -->

## Paper
This repository provides the official implementation of training SMART-Net as well as the usage of the pre-trained SMART-Net in the following paper:

<b>Improved performance and robustness of multi-task representation learning with consistency loss between pretexts for intracranial hemorrhage identification in head CT</b> <br/>
[Sunggu Kyung](https://github.com/babbu3682)<sup>1</sup>, Keewon Shina, Hyunsu Jeongb, Ki Duk Kimb, Jooyoung Parka, Kyungjin Choa, Jeong Hyun Leec, Gil-Sun Hongc, and Namkug Kim <br/>
<sup>1 </sup>Arizona State University,   <sup>2 </sup>Mayo Clinic <br/>
<b>(Under revision...)</b> Medical Image Analysis (MedIA) <br/>
<!-- [paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/babbu3682/SMART-Net) | [graphical abstract](https://ars.els-cdn.com/content/image/1-s2.0-S1361841520302048-fx1_lrg.jpg) -->
[code](https://github.com/babbu3682/SMART-Net)


## Requirements
+ Linux
+ Python 3.8.5
+ PyTorch 1.8.0


## SMART-Net Framework
### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/babbu3682/SMART-Net.git
$ cd SMART-Net/
$ pip install -r requirements.txt
```

### 2. Preparing data

#### For your convenience, we have provided a few 3D nii samples from AMC dataset as well as their mask labels.
Download the data from [this repository](https://zenodo.org/record/4625321/files/TransVW_data.zip?download=1). We have provided the training and validation samples for C=50 classes of visual words. For each instance of a visual word, we have extracted 3 multi-resolution cubes from each patient, where each of the three resolutions are saved in files named as 'train_dataN_vwGen_ex_ref_fold1.0.npy',  *N*=1,2,3. For each 'train_dataN_vwGen_ex_ref_fold1.0.npy' file, there is a corresponding 'train_labelN_vwGen_ex_ref_fold1.0.npy' file, which contains the pseudo labels of the discovered visual words.  


- The processed anatomical patterns directory structure
```
TransVW_data/
    |--  train_data1_vwGen_ex_ref_fold1.0.npy  : training data - resolution 1
    |--  train_data2_vwGen_ex_ref_fold1.0.npy  : training data - resolution 2
    |--  train_data3_vwGen_ex_ref_fold1.0.npy  : training data - resolution 3
    |--  val_data1_vwGen_ex_ref_fold1.0.npy    : validation data
    |--  train_label1_vwGen_ex_ref_fold1.0.npy : training labels - resolution 1
    |--  train_label2_vwGen_ex_ref_fold1.0.npy : training labels - resolution 2
    |--  train_label3_vwGen_ex_ref_fold1.0.npy : training labels - resolution 3
    |--  val_label1_vwGen_ex_ref_fold1.0.npy   : validation labels
   
```

### 3. Upstream
We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.


### 4. Downstream
We conducted downstream training using multi-task representation.


## Upstream visualize
### 1. Grad-CAM and activation map

### 2. t-SNE

## Excuse
For personal information security reasons of medical data, our data cannot be disclosed.


## Citation
If you use this code or use our pre-trained weights for your research, please cite our papers:
```
@InProceedings{zhou2019models,
  author="Zhou, Zongwei and Sodha, Vatsal and Rahman Siddiquee, Md Mahfuzur and Feng, Ruibin and Tajbakhsh, Nima and Gotway, Michael B. and Liang, Jianming",
  title="Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2019",
  year="2019",
  publisher="Springer International Publishing",
  address="Cham",
  pages="384--393",
  isbn="978-3-030-32251-9",
  url="https://link.springer.com/chapter/10.1007/978-3-030-32251-9_42"
}
```

## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the National Institutes of Health (NIH) under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. This is a patent-pending technology.
