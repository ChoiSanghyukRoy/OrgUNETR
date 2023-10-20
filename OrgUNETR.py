import torch
import torchvision as tv
import torch.nn as nn
from torch.nn import functional as F
import time
from PIL import Image
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from torchsummary import summary as model_summary
import glob
import nibabel as nib
import skimage
import sklearn
from sklearn import model_selection
import monai
import os 
import datetime as dt
from monai.config import print_config
from typing import Optional
from monai.metrics import MAEMetric
from tqdm import tqdm
import copy
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch import Tensor
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
    AsDiscrete,
    RandRotated,
    RandRotate,
)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
print()
monai.config.print_config()

### HYPER PARAMETER ###

RANDOM_SEED = 55
#IMAGE_SIZE = (256, 256, 256)
IMAGE_SIZE = (128, 128, 128)
BATCH_SIZE = 2
NUM_CLASS = 3
NUM_CLASS_ONE_HOT = 2
EPOCHS = 300
test_ratio, val_ratio = 0.2, 0.05

MODEL_SAVE = True
if MODEL_SAVE:
    model_dir1 = '/home/gail1/workspace/save_model/'
    model_dir2 = 'results'
    MODEL_SAVE_PATH = os.path.join(model_dir1, model_dir2)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


USE_MY_DATA = True

if not USE_MY_DATA:
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root dir is: {root_dir}")
    
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

else:
    # your dataset path
    data_dir = '/home/gail1/CSH/dataset/kits2023/'



train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_data_dicts, val_data_dicts = data_dicts[:-9], data_dicts[-9:]
train_data_dicts[0]



train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

TrainSet, TestSet = model_selection.train_test_split(data_dicts, test_size=test_ratio, random_state=RANDOM_SEED)
TrainSet, ValSet = model_selection.train_test_split(TrainSet, test_size=val_ratio, random_state=RANDOM_SEED)
print('TrainSet:', len(TrainSet), 'ValSet:', len(ValSet), 'TestSet:', len(TestSet))



for i in range(3):
    sample_img = nib.load(TrainSet[i]['image']).get_fdata()
    sample_mask = nib.load(TrainSet[i]['label']).get_fdata()
    print(f"[sample {i+1}] {os.path.basename(TrainSet[i]['image'])} {os.path.basename(TrainSet[i]['label'])}")
    print(sample_img.shape, sample_img.dtype, np.min(sample_img), np.max(sample_img))
    print(sample_mask.shape, sample_mask.dtype, np.unique(sample_mask))

from monai.transforms.compose import Transform, MapTransform

class MinMax(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] -= np.min(d[key])
            d[key] /= np.max(d[key])
        return d

loader = LoadImaged(keys=("image",'label'), image_only=False)
ensure_channel_first = EnsureChannelFirstd(keys=["image",'label'])
orientation = Orientationd(keys=["image",'label'], axcodes="RAS")
resize_img = Resized(keys=["image",], spatial_size=(IMAGE_SIZE), mode='trilinear')
resize_mask = Resized(keys=['label',], spatial_size=(IMAGE_SIZE), mode='nearest-exact')
minmax = MinMax(keys=['image',])



transforms = Compose([    
    LoadImaged(keys=("image",'label'), image_only=False),
    EnsureChannelFirstd(keys=["image",'label']),
    Orientationd(keys=["image",'label'], axcodes="RAS"),    
    MinMax(keys=['image',]),
    ToTensord(keys=["image", "label"]),
    RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.3, keep_size=True),
    ])

transforms_val = Compose([    
    LoadImaged(keys=("image",'label'), image_only=False),
    EnsureChannelFirstd(keys=["image",'label']),
    Orientationd(keys=["image",'label'], axcodes="RAS"),    
    MinMax(keys=['image',]),
    ToTensord(keys=["image", "label"]),    
    ])

SampleSet = transforms(TestSet[:3])

for i in range(3):
    sample_img = SampleSet[i]['image']
    sample_mask = SampleSet[i]['label']
    print(f"[sample {i+1}]")
    print(sample_img.shape, sample_img.dtype, torch.min(sample_img), torch.max(sample_img))
    print(sample_mask.shape, sample_mask.dtype, torch.unique(sample_mask))


'''
ncols, nrows = 10, 6
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/2))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2 = 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%2 == 0:
            axes[i,j].imshow(SampleSet[0]['image'][0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        else:
            axes[i,j].imshow(SampleSet[0]['label'][0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  
'''


train_ds = CacheDataset(
    data=TrainSet,
    transform=transforms,
    cache_num=4,
    cache_rate=1.0,
    num_workers=0)
val_ds = CacheDataset(
    data=ValSet, transform=transforms_val, cache_num=2, cache_rate=1.0, num_workers=0)
test_ds = CacheDataset(
    data=TestSet, transform=transforms_val, cache_num=2, cache_rate=1.0, num_workers=0)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
    
class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(            
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)            
        )

    def forward(self, x):        
        x = self.block(x)                        
        return x

class Embeddings(nn.Module):
    def __init__(self, input_shape, patch_size=16, embed_dim=768, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = input_shape[-4]
        self.n_patches = int((input_shape[-1] * input_shape[-2] * input_shape[-3]) / (patch_size * patch_size * patch_size))
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=self.in_channels, out_channels=self.embed_dim,
                                          kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = rearrange(x, "b n h w d -> b (h w d) n")
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings



class SEBlock(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        input_shape=(128, 128, 128)
        patch_size=16
        self.n_patches = int((input_shape[-1] * input_shape[-2] * input_shape[-3]) / (patch_size * patch_size * patch_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        out = self.qkv(x) + self.position_embeddings
        out = self.dropout(out)
        out = x*out
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Dropout(drop_p),
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.GELU()
        )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, depth=12, dropout=0., extract_layers=[3,6,9,12]):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, SEBlock(embed_dim, num_heads, dropout)),
                PreNorm(embed_dim, FeedForwardBlock(embed_dim, expansion=4))
            ]))            
        self.extract_layers = extract_layers   # <- list
        
    def forward(self, x):
        extract_layers = []
        
        for cnt, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if cnt+1 in self.extract_layers:
                extract_layers.append(x)
            
        return extract_layers




class UNETR(nn.Module):
    def __init__(self, img_shape=(224, 224, 224), input_dim=3, output_dim=3, 
                 embed_dim=768, patch_size=16, num_heads=8, dropout=0.1, light_r=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [2, 4, 6, 8]
        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.conv_channels = [int(i) for i in [16, 16, 16, 16, 16, 16]]
        self.conv_channels_out = int(self.conv_channels[0]/2)

        self.embedding = Embeddings((input_dim,*img_shape), embed_dim=self.embed_dim, dropout=self.dropout, patch_size = self.patch_size)
        
        self.transformer = \
            TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, extract_layers=self.ext_layers
            )
        
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, self.conv_channels[0], 3),
                Conv3DBlock(self.conv_channels[0], self.conv_channels_out, 3),
                nn.Dropout(self.dropout)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, self.conv_channels[2]),
                Deconv3DBlock(self.conv_channels[2], self.conv_channels[2]),
                Deconv3DBlock(self.conv_channels[2], self.conv_channels_out),
                nn.Dropout(self.dropout)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, self.conv_channels[3]),
                Deconv3DBlock(self.conv_channels[3], self.conv_channels_out),
                nn.Dropout(self.dropout)
            )

        self.decoder9 = \
            Deconv3DBlock(self.embed_dim, self.conv_channels_out)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(self.embed_dim, self.conv_channels_out)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[5], self.conv_channels[3]),
                Conv3DBlock(self.conv_channels[3], self.conv_channels[3]),
                Conv3DBlock(self.conv_channels[3], self.conv_channels[3]),
                SingleDeconv3DBlock(self.conv_channels[3], self.conv_channels_out),  # self.conv_channels_out = 16
                nn.Dropout(self.dropout),
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[4], self.conv_channels[2]),
                Conv3DBlock(self.conv_channels[2], self.conv_channels[2]),
                SingleDeconv3DBlock(self.conv_channels[2], self.conv_channels_out),
                nn.Dropout(self.dropout),
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[3], self.conv_channels[1]),
                Conv3DBlock(self.conv_channels[1], self.conv_channels[1]),
                SingleDeconv3DBlock(self.conv_channels[1], self.conv_channels_out),
                nn.Dropout(self.dropout),
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[2], self.conv_channels[1]),
                Conv3DBlock(self.conv_channels[1], self.conv_channels[1]),
                SingleConv3DBlock(self.conv_channels[1], output_dim, 1),
                nn.Dropout(self.dropout),
                #nn.Sigmoid()
            )
        
    def forward(self, x):
        z0 = x
        x = self.embedding(x)
        z = self.transformer(x)
        z2, z4, z6, z8 = z
        z2 = z2.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z4 = z4.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z8 = z8.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z8 = self.decoder12_upsampler(z8)
        z6 = self.decoder9(z6)
        z6 = self.decoder9_upsampler(torch.cat([z6, z8], dim=1))
        z4 = self.decoder6(z4)
        z4 = self.decoder6_upsampler(torch.cat([z4, z6], dim=1))
        z2 = self.decoder3(z2)
        z2 = self.decoder3_upsampler(torch.cat([z2, z4], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z2], dim=1))
        return output


from monai.losses import DiceCELoss, DiceLoss
from monai.losses.dice import one_hot

# model hyper-parameters
model = UNETR(img_shape=IMAGE_SIZE, input_dim=1, output_dim=4, 
              embed_dim=128, patch_size=16, num_heads=32, dropout=0.2, light_r=0.5)
model = model.to(DEVICE)
model_summary(model, (1,*IMAGE_SIZE), device=DEVICE.type)



torch.backends.cudnn.benchmark = True # ??

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5*5)
LossFunction = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
MetricDice = monai.metrics.DiceMetric(include_background=False, reduction="mean")



def BinaryOutput(output, keepdim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=NUM_CLASS_ONE_HOT)
    if keepdim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)
    
    return argmax_oh

print("done")

def train(epoch, train_loader):
    mean_epoch_loss = 0
    mean_dice_score_organ = 0
    mean_dice_score_tumor = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X EPOCHS) (loss=X.X) (dice score=%.5f)", 
        dynamic_ncols=True)

    data_check_iter = 0
    for step, batch in enumerate(epoch_iterator):

        x, y = (batch["image"].to(DEVICE), batch["label"])
        y_organ = torch.zeros(y.shape)
        y_organ[y==1] = 1
        y_organ[y==2] = 1
        y_organ = one_hot(y_organ, num_classes=NUM_CLASS_ONE_HOT)
        y_organ = y_organ.to(DEVICE)

        y_tumor = torch.zeros(y.shape)
        y_tumor[y==1] = 0
        y_tumor[y==2] = 1
        y_tumor = one_hot(y_tumor, num_classes=NUM_CLASS_ONE_HOT)
        y_tumor = y_tumor.to(DEVICE)

        pred = model(x)
        pred_organ = pred[:,0:2,:,:,:]     
        pred_tumor = pred[:,2:4,:,:,:]
        pred_organ = pred_organ.to(DEVICE)
        pred_tumor = pred_tumor.to(DEVICE)
        
        
        loss_organ = LossFunction(pred_organ, y_organ)
        loss_tumor = LossFunction(pred_tumor, y_tumor)
        
        loss = loss_organ * 0.68 + loss_tumor * 0.32 

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        mean_epoch_loss += loss.item()

        bi_organ = BinaryOutput(pred_organ)
        MetricDice(bi_organ, y_organ)
        dice_score_organ = MetricDice.aggregate().item()
        mean_dice_score_organ += dice_score_organ

        MetricDice.reset()

        bi_tumor = BinaryOutput(pred_tumor)
        MetricDice(bi_tumor, y_tumor)
        dice_score_tumor = MetricDice.aggregate().item()
        mean_dice_score_tumor += dice_score_tumor

        MetricDice.reset()
        
        epoch_iterator.set_description(
            "Training (%d / %d EPOCHS) (loss= %2.4f) (dice score organ=%.4f) (dice score tumor = %.4f)" 
            % (epoch, EPOCHS, loss.item(), dice_score_organ, dice_score_tumor))
        
    
    mean_epoch_loss /= len(epoch_iterator)    
    mean_dice_score_organ /= len(epoch_iterator)
    mean_dice_score_tumor /= len(epoch_iterator)

    print("mean epoch loss : ", mean_epoch_loss)
    print("mean dice score organ : ", mean_dice_score_organ)
    print("mean dice score tumor : ", mean_dice_score_tumor)

    return mean_epoch_loss, mean_dice_score_organ, mean_dice_score_tumor





def evaluate(epoch, test_loader):
    model.eval() 
    mean_epoch_loss = 0
    mean_dice_score_organ = 0
    mean_dice_score_tumor = 0
    epoch_iterator = tqdm(
        test_loader, desc="Evaluating (X / X EPOCHS) (loss=X.X) (dice score=%.5f)", 
        dynamic_ncols=True)
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].to(DEVICE), batch["label"])

            y_organ = torch.zeros(y.shape)
            y_organ[y==1] = 1
            y_organ[y==2] = 1
            y_organ = one_hot(y_organ, num_classes=NUM_CLASS_ONE_HOT)
            y_organ = y_organ.to(DEVICE)

            y_tumor = torch.zeros(y.shape)
            y_tumor[y==1] = 0
            y_tumor[y==2] = 1
            y_tumor = one_hot(y_tumor, num_classes=NUM_CLASS_ONE_HOT)
            y_tumor = y_tumor.to(DEVICE)

            pred = model(x)
            pred_organ = pred[:,0:2,:,:,:]        
            pred_tumor = pred[:,2:4,:,:,:]
            pred_organ = pred_organ.to(DEVICE)
            pred_tumor = pred_tumor.to(DEVICE)
            
            loss_organ = LossFunction(pred_organ, y_organ)
            loss_tumor = LossFunction(pred_tumor, y_tumor)
            
            loss = loss_organ * 0.68 + loss_tumor * 0.32
            
            mean_epoch_loss += loss.item()

            bi_organ = BinaryOutput(pred_organ)
            MetricDice(bi_organ, y_organ)
            dice_score_organ = MetricDice.aggregate().item()
            mean_dice_score_organ += dice_score_organ

            MetricDice.reset()

            bi_tumor = BinaryOutput(pred_tumor)
            MetricDice(bi_tumor, y_tumor)
            dice_score_tumor = MetricDice.aggregate().item()
            mean_dice_score_tumor += dice_score_tumor         

            MetricDice.reset()   
            
            epoch_iterator.set_description(
                "Evaluating (%d / %d EPOCHS) (loss= %2.4f) (dice organ=%.5f) (dice tumor = %.5f)" 
                % (epoch, EPOCHS, loss.item(), dice_score_organ, dice_score_tumor))

        mean_epoch_loss /= len(epoch_iterator)
        mean_dice_score_organ /= len(epoch_iterator)
        mean_dice_score_tumor /= len(epoch_iterator)

        print("mean epoch loss : ", mean_epoch_loss)
        print("mean dice score organ : ", mean_dice_score_organ)
        print("mean dice score tumor : ", mean_dice_score_tumor)
        
    return mean_epoch_loss, mean_dice_score_organ, mean_dice_score_tumor   





losses = {'train':[], 'val':[]}
dice_scores_organ = {'train':[], 'val':[]}
dice_scores_tumor = {'train':[], 'val':[]}
best_metric, best_epoch = -1, -1

iter = 0

for epoch in range(1, EPOCHS+1):
    train_loss, train_dice_score_organ, train_dice_score_tumor = train(epoch, train_loader)
    val_loss, val_dice_score_organ, val_dice_score_tumor = evaluate(epoch, val_loader)
    losses['train'].append(train_loss)
    losses['val'].append(val_loss)
    dice_scores_organ['train'].append(train_dice_score_organ)
    dice_scores_organ['val'].append(val_dice_score_organ)
    dice_scores_tumor['train'].append(train_dice_score_tumor)
    dice_scores_tumor['val'].append(val_dice_score_tumor)

    if dice_scores_tumor['val'][-1] > best_metric:
        if epoch > 100:
            best_metric = dice_scores_tumor['val'][-1]
            best_epoch = epoch
            print(f'Best record! [{epoch}] Test Loss: {val_loss:.6f}, Dice organ : {val_dice_score_organ:.6f}, Dice tumor : {val_dice_score_tumor:.6f}')
            if MODEL_SAVE:
                model_name = f'./{best_epoch}_{best_metric}.pth'
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))
                print('saved model')
            
with open("mean_losses_train.txt", "w") as output:
    for item in losses['train']:
        output.write("%f\n" % item)

with open("mean_losses_val.txt", "w") as output:
    for item in losses['val']:
        output.write("%f\n" % item)
    
with open("mean_dice_score_organ_train.txt", "w") as output:
    for item in dice_scores_organ['train']:
        output.write("%f\n" % item)
    
with open("mean_dice_score_organ_val.txt", "w") as output:
    for item in dice_scores_organ['val']:
        output.write("%f\n" % item)
    
with open("mean_dice_score_tumor_train.txt", "w") as output:
    for item in dice_scores_tumor['train']:
        output.write("%f\n" % item)

with open("mean_dice_score_tumor_val.txt", "w") as output:
    for item in dice_scores_tumor['val']:
        output.write("%f\n" % item)
    

"""
epochs = [i for i in range(len(losses['train']))]
train_loss = losses['train']
val_loss = losses['val']
train_dice_organ = dice_scores_organ['train']
val_dice_organ = dice_scores_organ['val']
train_dice_tumor = dice_scores_tumor['train']
val_dice_tumor = dice_scores_tumor['val']

fig , ax = plt.subplots(1,3)
fig.set_size_inches(18,6)

ax[0].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[0].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(epochs , train_dice_organ , 'go-' , label = 'Training Dice score organ')
ax[1].plot(epochs , val_dice_organ , 'ro-' , label = 'Validation Dice score organ')
ax[1].set_title('Training & Validation Dice score for organ')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Dice score organ")

ax[2].plot(epochs , train_dice_tumor , 'go-' , label = 'Training Dice score tumor')
ax[2].plot(epochs , val_dice_tumor , 'ro-' , label = 'Validation Dice score tumor')
ax[2].set_title('Training & Validation Dice score for tumor')
ax[2].legend()
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Dice score tumor")

plt.show()
plt.savefig('results_with_small_size.png')


pred_dict = {'input':[], 'target':[], 'output_organ':[], 'output_tumor':[]}

if MODEL_SAVE:
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, model_name)))

model.to('cpu')
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, target = data["image"].cpu(), data["label"].cpu()

        output = model(img).detach().cpu()
        output_organ = output[:,0:2,:,:,:]
        output_tumor = output[:,2:4,:,:,:]
        output_organ = torch.argmax(output_organ, dim=1)
        output_tumor = torch.argmax(output_tumor, dim=1)
        
        pred_dict['input'].append(img)
        pred_dict['target'].append(target)
        pred_dict['output_organ'].append(output_organ)
        pred_dict['output_tumor'].append(output_tumor)

        numpy_target = target.numpy()
        filename = 'target_patch_4_' + str(i) + ".npy"
        np.save(filename, numpy_target)
        numpy_img = img.numpy()
        filename = 'image_patch_4_' + str(i) + ".npy"
        np.save(filename, numpy_img)

        numpy_organ = output_organ.numpy()
        filename = 'organ_prediction_patch_4_' + str(i) + ".npy"
        np.save(filename, numpy_organ)
        numpy_tumor = output_tumor.numpy()
        filename = 'tumor_prediction_patch_4_' + str(i) + ".npy"
        np.save(filename, numpy_tumor)
        
        #if i > 10:
            #break


ncols, nrows = 10, 3*3
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/3))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2, cnt3 = 0, 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%3 == 0:
            axes[i,j].imshow(pred_dict['input'][0][0,0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        elif i%3 == 1:
            axes[i,j].imshow(pred_dict['target'][0][0,0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        else:
            axes[i,j].imshow(pred_dict['output_organ'][0][0,:,:,cnt3], cmap='gray')
            cnt3+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  
plt.savefig('predictions_organ_transformer_X.png')



ncols, nrows = 10, 3*3
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/3))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2, cnt3 = 0, 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%3 == 0:
            axes[i,j].imshow(pred_dict['input'][0][0,0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        elif i%3 == 1:
            axes[i,j].imshow(pred_dict['target'][0][0,0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        else:
            axes[i,j].imshow(pred_dict['output_tumor'][0][0,:,:,cnt3], cmap='gray')
            cnt3+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  
plt.savefig('predictions_tumor_transformer_X.png')
"""