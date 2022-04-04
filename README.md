# SMART-Net - Official Pytorch Implementation

It's scheduled to be uploaded soon. We are modifying the code for easier use.
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
Highlights


<p align="center"><img width="100%" src="figures/graphical_abstract.png" /></p>
<!-- <p align="center"><img width="85%" src="figures/framework.png" /></p> -->

## Paper
This repository provides the official implementation of training SMART-Net as well as the usage of the pre-trained SMART-Net in the following paper:

<b>Improved performance and robustness of multi-task representation learning with consistency loss between pretexts for intracranial hemorrhage identification in head CT</b> <br/>
[Sunggu Kyung](https://github.com/babbu3682)<sup>1</sup>, Keewon Shina, Hyunsu Jeongb, Ki Duk Kimb, Jooyoung Parka, Kyungjin Choa, Jeong Hyun Leec, Gil-Sun Hongc, and Namkug Kim <br/>
[MI2RL LAB](https://www.mi2rl.co/) <br/>
<b>(Under revision...)</b> Medical Image Analysis (MedIA) <br/>
<!-- [paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/babbu3682/SMART-Net) | [graphical abstract](https://ars.els-cdn.com/content/image/1-s2.0-S1361841520302048-fx1_lrg.jpg) -->
[code](https://github.com/babbu3682/SMART-Net)


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
You can use your own data using the [dicom2nift](https://github.com/icometrix/dicom2nifti) for converting from dicom to nii.

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

### 3. Upstream

#### 📋 Available List
- [x] Up_SMART_Net
- [x] Up_SMART_Net_Dual_CLS_SEG
- [x] Up_SMART_Net_Dual_CLS_REC
- [x] Up_SMART_Net_Dual_SEG_REC
- [x] Up_SMART_Net_Single_CLS
- [x] Up_SMART_Net_Single_SEG
- [x] Up_SMART_Net_Single_REC


**+ train**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python train.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--model-name 'Up_SMART_Net' \
--batch-size 10 \
--epochs 1000 \
--num-workers 4 \
--pin-mem \
--training-stream 'Upstream' \
--multi-gpu-mode 'DataParallel' \
--cuda-visible-devices '2, 3' \
--gradual-unfreeze 'True' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test'
```
**+ test (slice-wise for slice-level)**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python test.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "True" \
--model-name 'Up_SMART_Net' \
--num-workers 4 \
--pin-mem \
--training-stream 'Upstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test/epoch_0_checkpoint.pth'
```
**+ test (stacking slice for volume-level)**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python test.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "False" \
--model-name 'Up_SMART_Net' \
--num-workers 4 \
--pin-mem \
--training-stream 'Upstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test/epoch_0_checkpoint.pth'
```

### 4. Downstream

#### 📋 Available List
- [x] Down_SMART_Net_CLS
- [x] Down_SMART_Net_SEG

#### - SMART-Net-CLS
**+ train**: We conducted downstream training using multi-task representation.
```bash
python train.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--model-name 'Down_SMART_Net_CLS' \
--batch-size 2 \
--epochs 1000 \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'DataParallel' \
--cuda-visible-devices '2, 3' \
--gradual-unfreeze 'True' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_cls_test' \
--from-pretrained '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/[UpTASK]ResNet50_ImageNet.pth' \
--load-weight-type 'encoder'
```
**+ test**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python test.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner 'False' \
--model-name 'Down_SMART_Net_CLS' \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_cls_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_cls_test/epoch_0_checkpoint.pth'

```

#### - SMART-Net-SEG
**+ train**: We conducted downstream training using multi-task representation.
```bash
python train.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--model-name 'Down_SMART_Net_SEG' \
--batch-size 2 \
--epochs 1000 \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'DataParallel' \
--cuda-visible-devices '2, 3' \
--gradual-unfreeze 'True' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_seg_test' \
--from-pretrained '/workspace/sunggu/1.Hemorrhage/SMART-Net/up_test/epoch_0_checkpoint.pth' \
--load-weight-type 'encoder'
```
**+ test**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python test.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner 'False' \
--model-name 'Down_SMART_Net_SEG' \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_seg_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_seg_test/epoch_0_checkpoint.pth'

```

## SMART-Net Inference
#### 📋 Available List
- [x] Up_SMART_Net
- [x] Down_SMART_Net_CLS
- [x] Down_SMART_Net_SEG

#### - Up-SMART-Net
**+ inference**: We conducted downstream training using multi-task representation.
```bash
python inference.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "True" \
--model-name 'Up_SMART_Net' \
--num-workers 4 \
--pin-mem \
--training-stream 'Upstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test/epoch_0_checkpoint.pth'

```
#### - Down-SMART-Net-CLS
**+ inference**: We conducted downstream training using multi-task representation.
```bash
python inference.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "False" \
--model-name 'Down_SMART_Net_CLS' \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_cls_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_cls_test/epoch_0_checkpoint.pth'

```
#### - Down-SMART-Net-SEG
**+ inference**: We conducted downstream training using multi-task representation.
```bash
python inference.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "False" \
--model-name 'Down_SMART_Net_SEG' \
--num-workers 4 \
--pin-mem \
--training-stream 'Downstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_seg_test/pred_nii' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/down_seg_test/epoch_0_checkpoint.pth'

```


## Upstream visualize
### 1. Activation map
```
⏳ It's scheduled to be uploaded soon.
```
### 2. t-SNE
```
⏳ It's scheduled to be uploaded soon.
```


## Excuse
For personal information security reasons of medical data in Korea, our data cannot be disclosed.


## 📝 Citation
If you use this code for your research, please cite our papers:
```
⏳ It's scheduled to be uploaded soon.
```

## 🤝 Acknowledgement
We build SMART-Net framework by referring to the released code at [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) and [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI). 
This is a patent-pending technology.


### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/babbu3682/SMART-Net/blob/main/LICENSE)