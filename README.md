### OrgUNETR

This repository contains code for OrgUNETR.
OrgUNETR, a segmentation model analyzing 3D CT scans, demonstrates the capacity to **detecting organ and tumor simultaneously**. OrgUNETR has two distinct output channels, one for the prediction of organ and the other for the prediction of tumor. Training without organ information fails to detect tumors. However, OrgUNETR trained with organ and tumor information successfully detects both organ and tumor.


### Dataset

We trained our model with KiTS2019 dataset containing CT scans. In the CT scan, the location of organ and tumor is annotated. The actual dimenison of the dataset is (512, 512, X) and X are arbitrary number between 29 to 1059. We resized our datasset into (128, 128, 128) uniformly.


### Architecture

The architecture of OrgUNETR is shown below. For details about OrgUNETR, please have a look at our python code.

![image](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/5feba058-530e-4c0e-918b-f18cb9dcd762)

OrgUNETR consists of two components which are patch embedding layer and **Squeeze and Excitation Block(SEBlock)**. The patch embedding layer, which is typically used in vision models, compresses CT scan into certain dimension. We replaced transformer layer in UNETR to Squeeze and Excitation layer evaluating attention score between patch sequences while maintaining accuracy.


### To run code,
The dataset path should be specified in line 108. The hyper-parameter of OrgUNETR can be modified in line 488.


### Contact
Feel free to send an email to ask questions about our model, OrgUNETR. Also, please don't hesitate to report problems with our code! 😆
## Contact Email : Choi Sanghyuk Roy (choiroy@cau.ac.kr)


