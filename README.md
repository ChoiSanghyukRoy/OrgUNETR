### OrgUNETR

This repository contains code for OrgUNETR.
OrgUNETR is a segmentation model that operates on 3D CT scans, detecting organ and tumor simultaneously. OrgUNETR has two output channel to predict organ and tumor separately. Training with only tumor information fails to detect tumors. However, OrgUNETR trained with organ and tumor information can detect both organ and tumor. We trained our model with KiTS2019 dataset which contains CT scans which the location of organ and tumor is annotated. 

The architecture of OrgUNETR is located below. For details about OrgUNETR, please have a look at our python code.
![image](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/5feba058-530e-4c0e-918b-f18cb9dcd762)
