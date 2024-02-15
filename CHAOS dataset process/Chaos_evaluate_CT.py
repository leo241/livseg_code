import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
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
from evaluate_metrics import dice_dir1_dir2

def save_arr2nii(arr, path='tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)

def chaos_label_liver_CT(label):  # 将label（512，512）转化为标准的mask形式（512，512，class_num）
    mask_shape = label.shape
    mask = np.zeros(mask_shape)
    mask[(label >= 55)] = 1  # foreground
    return mask

def dice_score(fig1, fig2, class_value): # dice系数，越大越好
    '''
    计算某种特定像素级类别的DICE SCORE
    :param fig1:
    :param fig2:
    :param class_value:
    :return:
    '''
    fig1_class = fig1 == class_value
    fig2_class = fig2 == class_value
    A = np.sum(fig1_class)
    B = np.sum(fig2_class)
    AB = np.sum(fig1_class & fig2_class)
    if A + B == 0:
        return 1
    return 2 * AB / (A + B)

def dice_dir1_dir2(dir1,dir2):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(dir1))  # ground truth
    gt = chaos_label_liver_CT(gt)
    pre = sitk.GetArrayFromImage(sitk.ReadImage(dir2))
    ans = dice_score(gt, pre, 1)
    # print(ans)
    return ans

# gt_list_dir = []
# tmp = os.listdir(r'D:\study\pga\newtry0427\CHAOS_Train_Sets\label')
# for item in tmp:
#     if 'T2' in item:
#         gt_list_dir.append(item)
# name_lst = []
# dst = []
# for label1 in tqdm(sorted(gt_list_dir)):
#     label1_lst = label1.split('.')
#     label1_lst[0] = label1_lst[0] + '_label'
#     label2 = '.'.join(label1_lst)
#     nii_dir1 = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\label\{label1}'
#     nii_dir2 = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\bayeseg\{label2}'
#     dice = dice_dir1_dir2(nii_dir1,nii_dir2)
#     name_lst.append(label1)
#     dst.append(dice)
# # for i in name_lst:
# #     print(i)
# for i in dst:
#     print(i)

if __name__ == '__main__':
    nii_dir1 = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\label_CT\16.nii.gz'
    nii_dir2 = fr'D:\study\pga\newtry0427\UI design\16_label.nii.gz'
    dice = dice_dir1_dir2(nii_dir1,nii_dir2)
    print(dice)