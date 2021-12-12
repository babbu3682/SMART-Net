# MRI-Net
It's scheduled to be uploaded soon. We are modifying the code for easier use.


![image](https://user-images.githubusercontent.com/48959435/145704189-f68032cd-481f-4395-998f-877836df5f8e.png)
Fig. 2. Schematic overview of MRI-Net. Our model is composed of upstream and downstream tasks; a) A upstream task is a 2D-based representation learning process using MTL (classification, segmentation, reconstruction, and consistency). A downstream task is a process of extending to volume-level ICH tasks by the transfer learning using the pre-trained encoder in the upstream. b) A pre-trained feature extractor and an LSTM-based classifier are combined to perform volume-level classification. c) The pre-trained feature extractor and the Conv3D-based segmentor are combined to perform volume-level segmentation.

![image](https://user-images.githubusercontent.com/48959435/145704193-45adb8e9-db02-4f14-8ec0-da786489e7ec.png)
Fig. 4. Comparisons of the encoder activation map in the upstream task according to multi-task combinations and the previous representation learning approaches. In order to perform a comparison according to the severity of ICH cases, two severe cases (≥ 30 mL), two moderate cases (15-30 mL), and four mild cases (≤ 15 mL) of ICH were compared.
