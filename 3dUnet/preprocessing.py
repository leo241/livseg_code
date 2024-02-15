import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import transform
from dataset_divide import single_domain_dataset,domain_generalization_dataset # 数据集划分
# from unet import UNet # 网络结构
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标
import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

# 超参数设置
 # 图片统一resize的大小
def resize_3d_image(image, size = (32,120,120)):
    image =transform.resize(image,size, anti_aliasing=False, preserve_range=True) # 这两个参数的设置非常重要，抗锯齿，且在插值时保持数据值的大小
    return image


def get_data_preprocess(trains,vals,tests): # 图片获取和预处理（01缩放+resize）
    DIR = 'D:/study/pga/dataset/mydata2' # 储存数据的绝对路径
    train_list = list()
    for dirname in trains:
        img = sitk.ReadImage(f'{DIR}/image/{dirname}')
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(tuple(sorted(img.shape))) # 从小到大顺序
        if len(img.shape) == 4: # DWI高B值
            img = img[0]
        minimum = np.min(img)
        gap = np.max(img) - minimum
        img = (img - minimum) / gap * 255  # 0，1缩放
        label = sitk.ReadImage(f'{DIR}/label/{dirname}')
        label = sitk.GetArrayFromImage(label)
        label = label.reshape(tuple(sorted(label.shape)))# 从小到大顺序
        img = resize_3d_image(img) # resize
        label = resize_3d_image(label) # resize
        train_list.append([img, label])

    val_list = list()
    for dirname in vals:
        img = sitk.ReadImage(f'{DIR}/image/{dirname}')
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(tuple(sorted(img.shape)))  # 从小到大顺序
        if len(img.shape) == 4:  # DWI高B值
            img = img[0]
        minimum = np.min(img)
        gap = np.max(img) - minimum
        img = (img - minimum) / gap * 255  # 0，1缩放
        label = sitk.ReadImage(f'{DIR}/label/{dirname}')
        label = sitk.GetArrayFromImage(label)
        label = label.reshape(tuple(sorted(label.shape)))  # 从小到大顺序
        img = resize_3d_image(img)  # resize
        label = resize_3d_image(label)  # resize
        val_list.append([img, label])

    test_list = list()
    for dirname in tests:
        img = sitk.ReadImage(f'{DIR}/image/{dirname}')
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(tuple(sorted(img.shape)))  # 从小到大顺序
        if len(img.shape) == 4:  # DWI高B值
            img = img[0]
        minimum = np.min(img)
        gap = np.max(img) - minimum
        img = (img - minimum) / gap * 255  # 0，1缩放
        label = sitk.ReadImage(f'{DIR}/label/{dirname}')
        label = sitk.GetArrayFromImage(label)
        label = label.reshape(tuple(sorted(label.shape)))  # 从小到大顺序
        img = resize_3d_image(img)  # resize
        label = resize_3d_image(label)  # resize
        test_list.append([img, label])
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
        img, mask = self.data[item]
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.data)