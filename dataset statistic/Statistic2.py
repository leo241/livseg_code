import numpy as np
import SimpleITK as sitk
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
from collections import Counter
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分


# def path2info(path):
#     img = sitk.ReadImage(path)
#     spacing = img.GetSpacing()
#     x_spacing,y_spacing,z_spacing = spacing[0],spacing[1],spacing[2]
#     slice_num = img.GetDepth()
#     return x_spacing,y_spacing,z_spacing,slice_num
#
# def statistic_info(domain):
#     domain_paths = single_domain_paths(domain)
#     x_spacings = []
#     y_spacings = []
#     z_spacings = []
#     slice_nums = []
#     for path in tqdm(domain_paths):
#         # print(path)
#         x_spacing, y_spacing, z_spacing, slice_num = path2info(path)
#         x_spacings.append(x_spacing)
#         y_spacings.append(y_spacing)
#         z_spacings.append(z_spacing)
#         slice_nums.append(slice_num)
#     print('slice nums range',np.min(slice_nums), np.max(slice_nums))
#     print('slice nums amount',sum(slice_nums))
#     print('x spacing range',np.min(x_spacings), np.max(x_spacings))
#     print('y spacing range',np.min(y_spacings), np.max(y_spacings))
#     print('z spacing range',np.min(z_spacings), np.max(z_spacings))

def Statistic_info(dataset):
    DIR = 'D:/study/pga/dataset/mydata2'  # 储存数据的绝对路径
    i = 0
    for path in dataset:
        img = sitk.ReadImage(f'{DIR}/image/{path}')
        slice_num = img.GetDepth()
        i += slice_num
    return len(dataset), i

if __name__ == '__main__':
    trains, vals, tests = domain_generalization_dataset('lianying','tongyong',ratio=[1,1])
    print(Statistic_info(trains))
    print(Statistic_info(vals))
    print(Statistic_info(tests))

