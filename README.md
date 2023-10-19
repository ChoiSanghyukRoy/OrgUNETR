### OrgUNETR

This repository contains code for OrgUNETR.
OrgUNETR is a segmentation model that operates on 3D CT scans, detecting organ and tumor simultaneously. OrgUNETR has two output channel to predict organ and tumor separately. Training with only tumor information fails to detect tumors. However, OrgUNETR trained with organ and tumor information can detect both organ and tumor.


### Dataset

We trained our model with KiTS2019 dataset which contains CT scans which the location of organ and tumor is annotated. The actual dimenison of the dataset is (512, 512, X) and X are arbitrary number between 29 to 1059. We resized our datasset into (128, 128, 128) due to limited computational resource.


### Architecture

The architecture of OrgUNETR is located below. For details about OrgUNETR, please have a look at our python code.

![image](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/5feba058-530e-4c0e-918b-f18cb9dcd762)

OrgUNETR consists of two components which are patch embedding layer and Squeeze and Excitation Block(SEBlock). The patch embedding layer that is conventionally used in vision models compresses CT scan into certain dimension. We replaced multi-head self attention layer in UNETR with Squeeze and Excitation layer. With this modification, we reduced computational complexity by 13.9% while maintaining accuracy.


## To run our model,
The dataset path should be specified in line 108. The hyper-parameter of OrgUNETR can be modified in line 488.


### Contact
Feel free to send an email to ask questions about our model, OrgUNETR. Also, please don't hesitate to report problems with our code! 😆
## Contact Email : Choi Sanghyuk Roy (choiroy@cau.ac.kr)


