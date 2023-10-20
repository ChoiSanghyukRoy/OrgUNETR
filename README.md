## OrgUNETR

This repository contains code for OrgUNETR.
OrgUNETR, a segmentation model analyzing 3D CT scans, demonstrates the capacity to **detecting organ and tumor simultaneously**. OrgUNETR has two distinct output channels, one for the prediction of organ and the other for the prediction of tumor. Training without organ information fails to detect tumors. However, OrgUNETR trained with organ and tumor information successfully detects both organ and tumor.




## Dataset

We trained our model with kidney organ & tumor dataset(KiTS2019) and prostate organ & tumor dataset(Prostate158) containing CT scans. In the CT scan, the location of organ and tumor is annotated. The actual dimenison of the dataset is (512, 512, X) and X are arbitrary number between 29 to 1059. We resized our datasset into (128, 128, 128) uniformly.




## Architecture

The architecture of OrgUNETR is shown below. For details about OrgUNETR, please have a look at our python code.

![OrgUNETR_figure](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/ef584108-ca9b-432f-ad1b-4ce9c0847e05)


OrgUNETR consists of two components which are patch embedding layer and **Squeeze and Excitation Block(SEBlock)**. The patch embedding layer, which is typically used in vision models, compresses CT scan into certain dimension. We replaced transformer layer in UNETR to Squeeze and Excitation layer evaluating attention score between patch sequences while maintaining accuracy.




## Performance

![Dice Score for Github](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/d664a5f3-2208-4b4b-9c70-2a3181af9ded)

We evaluated the performance of OrgUNETR using KiTS2019 and Prostate158 dataset. The dice score is used for performance comparison. The results are shown below. The graph on the right is a performance evaluation of OrgUNETR with KiTS2019. Additionally, a performance evaluation with a Prostate158 dataset is shown on the left side of the figure. Both graph showed that training with organ and tumor information helps OrgUNETR to detect tumors.




## Prediction Examples

We conducted experiments and the results are shown below

![Prediction Examples for Github](https://github.com/ChoiSanghyukRoy/OrgUNETR/assets/148459212/670558e1-10ba-4895-851c-665878e9b7c0)

Right figure is the prediction against kidney organ and tumor and left figure is the prediction against prostrate organ and tumor.




## To run code,
The dataset path should be specified in line 108. The hyper-parameter of OrgUNETR can be modified through line 475.




## Contact
Feel free to send an email to ask questions about our model, OrgUNETR. Also, please don't hesitate to report problems with our code! 😆
### Contact Email : Choi Sanghyuk Roy (choiroy@cau.ac.kr)


