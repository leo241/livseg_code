# 本文件的作用为查看预测样本和真实的dice score

import random

import pandas as pd
import numpy as np
import nibabel as nib  # 处理.nii类型图片
# import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from random import randint

# from unet import UNet # 同路径下的网络py文件
from PIL.PngImagePlugin import PngImageFile
import warnings

import surface_distance as surfdist # 该库从https://github.com/deepmind/surface-distance下载

def RAVD(Vref, Vseg): # 相对绝对体积差，结果是比例，越小越好
    ravd = (abs(Vref.sum() - Vseg.sum()) / Vref.sum())
    return ravd

warnings.filterwarnings("ignore")  # ignore warnings



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

def assd(gt,pre): # mm级别，平均对称表面距离，越小越好
    surface_distances = surfdist.compute_surface_distances(gt.astype(bool), pre.astype(bool),
                                                           spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    return np.mean(avg_surf_dist)/10


if __name__ == '__main__':
    biz = 'feilipu'
    level = 'S3'
    person = '171_ligenfa_P'
    MRI_type = 'DWI'
    gold_standard = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\{MRI_type}.nii.gz'
    predict = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\pre_{MRI_type}.nii.gz'
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gold_standard))  # ground truth
    pre = sitk.GetArrayFromImage(sitk.ReadImage(predict))
    print(dice_score(gt, pre, 1))
    print(RAVD(gt,pre))
    print(assd(gt,pre))
