# SMART-Net - Official Pytorch Implementation

We proposed a supervised multi-task aiding representation transfer learning network called <b>SMART-Net</b>.


## 💡 Highlights
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


<p align="center"><img width="100%" src="figures/graphical_abstract.png" /></p>
<!-- <p align="center"><img width="85%" src="figures/framework.png" /></p> -->

## Paper
This repository provides the official implementation of training SMART-Net as well as the usage of the pre-trained SMART-Net in the following paper:

<b>Improved performance and robustness of multi-task representation learning with consistency loss between pretexts for intracranial hemorrhage identification in head CT</b> <br/>
[Sunggu Kyung](https://github.com/babbu3682)<sup>1</sup>, Keewon Shin, Hyunsu Jeong, Ki Duk Kim, Jooyoung Park, Kyungjin Cho, Jeong Hyun Lee, Gil-Sun Hong, and Namkug Kim <br/>
[MI2RL LAB](https://www.mi2rl.co/) <br/>
<b>Journal: Medical Image Analysis (MedIA) <br/>

<font size=3><div align='center' > [**Paper**](https://www.sciencedirect.com/science/article/pii/S1361841522001360) | [**Code**](https://github.com/babbu3682/SMART-Net)</div></font>


## Requirements
+ Linux
+ Python 3.8.5
+ PyTorch 1.8.0


## 📦 SMART-Net Framework
### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/babbu3682/SMART-Net.git
$ cd SMART-Net/
$ pip install -r requirements.txt
```

### 2. Preparing data
#### For your convenience, we have provided few 3D nii samples from [Physionet publish dataset](https://physionet.org/content/ct-ich/1.3.1/) as well as their mask labels. 
#### Note: We do not use this data as a train, it is just for code publishing examples.

<!-- Download the data from [this repository](https://zenodo.org/record/4625321/files/TransVW_data.zip?download=1).  -->
You can use your own data using the [dicom2nifti](https://github.com/icometrix/dicom2nifti) for converting from dicom to nii.

- The processed hemorrhage directory structure
```
datasets/samples/
    train
        |--  sample1_hemo_img.nii.gz
        |--  sample1_hemo_mask.nii
        |--  sample2_normal_img.nii.gz
        |--  sample2_normal_mask.nii        
                .
                .
                .
    valid
        |--  sample9_hemo_img.nii.gz
        |--  sample9_hemo_mask.nii
        |--  sample10_normal_img.nii.gz
        |--  sample10_normal_mask.nii
                .
                .
                .
    test
        |--  sample20_hemo_img.nii.gz
        |--  sample20_hemo_mask.nii
        |--  sample21_normal_img.nii.gz
        |--  sample21_normal_mask.nii
                .
                .
                .   
```





## Excuse
For personal information security reasons of medical data in Korea, our data cannot be disclosed.


## 📝 Citation
If you use this code for your research, please cite our papers:
```BibTeX
@article{
  title={Improved performance and robustness of multi-task representation learning with consistency loss between pretexts for intracranial hemorrhage identification in head CT},
  author={Sunggu Kyung, Keewon Shin, Hyunsu Jeong, Ki Duk Kim, Jooyoung Park, Kyungjin Cho, Jeong Hyun Lee, Gil-Sun Hong, Namkug Kim},
  journal={Medical Image Analysis},
  year={2022}
}
```



## 🤝 Acknowledgement
We build SMART-Net framework by referring to the released code at [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) and [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI). 
This is a patent-pending technology.


### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/babbu3682/SMART-Net/blob/main/LICENSE)