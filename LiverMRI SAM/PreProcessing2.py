import numpy as np
import SimpleITK as sitk
from skimage import io,transform
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset_divide import single_domain_dataset,domain_generalization_dataset # 数据集划分
from unet import UNet # 网络结构
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

# 超参数设置
 # 图片统一resize的大小

class DataProcessor:
    def __init__(self,PIXEL = 256):
        self.background = 0
        self.pixel = PIXEL
        self.slice_resize = (self.pixel, self.pixel)

    def mask_one_hot(self,
                     img_arr):  # 将label（512，512）转化为标准的mask形式（512，512，class_num）
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1), 注意这里传进来的不是img，而是label
        mask_shape = img_arr.shape
        mask1 = np.zeros(mask_shape)
        mask2 = np.zeros(mask_shape)
        mask1[img_arr > self.background] = 1  # foreground
        mask2[img_arr == self.background] = 1
        mask = np.concatenate([mask1, mask2], 2)
        return mask
    def get_data(self,trains,vals,tests): # 图片获取和预处理（01缩放+resize）
        DIR = 'D:/study/pga/dataset/mydata2' # 储存数据的绝对路径

        train_list = list()
        for dirname in tqdm(trains):
            img = sitk.ReadImage(f'{DIR}/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            if len(img.shape) == 4: # DWI高B值
                img = img[0]
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'{DIR}/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img_np = img[id, :, :]
                # img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                # H, W, _ = img_3c.shape
                # img_1024 = transform.resize(
                #     img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                # ).astype(np.uint8)
                # img_1024 = (img_1024 - img_1024.min()) / np.clip(
                #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                # )
                # img_1024_tensor = (
                #     torch.tensor(img_1024).float().permute(2, 0, 1)
                # )
                label1 = label[id, :, :]

                # label1 = Image.fromarray(label1).convert('L')
                # label_resize = label1.resize((256,256), 0)
                # label_resize = np.asarray(label_resize)
                # label_resize = np.expand_dims(label_resize, 0)
                # label_resize = torch.tensor(label_resize).to(torch.float)
                train_list.append([img_np, label1, id, biz_type, person_name, MRI_type])

        val_list = list()
        for dirname in tqdm(vals):
            img = sitk.ReadImage(f'{DIR}/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            if len(img.shape) == 4:  # DWI高B值
                img = img[0]
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'{DIR}/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img_np = img[id, :, :]
                # img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                # H, W, _ = img_3c.shape
                # img_1024 = transform.resize(
                #     img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                # ).astype(np.uint8)
                # img_1024 = (img_1024 - img_1024.min()) / np.clip(
                #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                # )
                # img_1024_tensor = (
                #     torch.tensor(img_1024).float().permute(2, 0, 1)
                # )
                label1 = label[id, :, :]

                # label1 = Image.fromarray(label1).convert('L')
                # label_resize = label1.resize((256, 256), 0)
                # label_resize = np.asarray(label_resize)
                # label_resize = np.expand_dims(label_resize, 0)
                # label_resize = torch.tensor(label_resize).to(torch.float)
                val_list.append([img_np, label1, id, biz_type, person_name, MRI_type])

        test_list = list()
        for dirname in tqdm(tests):
            img = sitk.ReadImage(f'{DIR}/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            if len(img.shape) == 4:  # DWI高B值
                img = img[0]
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'{DIR}/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img_np = img[id, :, :]
                # img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                # img_1024 = transform.resize(
                #     img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                # ).astype(np.uint8)
                # img_1024 = (img_1024 - img_1024.min()) / np.clip(
                #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                # )
                # img_1024_tensor = (
                #     torch.tensor(img_1024).float().permute(2, 0, 1)
                # )
                label1 = label[id, :, :]

                # label1 = Image.fromarray(label1).convert('L')
                # label_resize = label1.resize((256, 256), 0)
                # label_resize = np.asarray(label_resize)
                # label_resize = np.expand_dims(label_resize, 0)
                # label_resize = torch.tensor(label_resize).to(torch.float)
                test_list.append([img_np, label1, id, biz_type, person_name, MRI_type])
        return train_list, val_list, test_list


class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data):
        self.data = data
        self.TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img_np, label1, id, biz_type, person_name, MRI_type = self.data[item]
        # img
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1)
        )
        # label
        label1 = Image.fromarray(label1).convert('L')
        label_resize = label1.resize((256, 256), 0)
        label_resize = np.asarray(label_resize)
        label_resize = np.expand_dims(label_resize, 0)
        label_resize = torch.tensor(label_resize).to(torch.float)

        return img_1024_tensor, label_resize, torch.tensor(id)

    def __len__(self):
        return len(self.data)