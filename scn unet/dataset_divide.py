'''
本文件的作用负责划分数据集，有单个域划分，也有域泛化的数据集划分
'''

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
from collections import Counter

# from unet import UNet # 同路径下的网络py文件
from PIL.PngImagePlugin import PngImageFile
import warnings

import surface_distance as surfdist # 该库从https://github.com/deepmind/surface-distance下载

def cross_modality_dataset(ratio=[3, 1, 1]):
    '''
    单独域的数据集划分
    '''
    paths = os.listdir('D:/study/pga/dataset/mydata2/image')
    random.seed(2023)
    # domain = 'T1'
    # ratio = [8,1,1] # 划分比例
    random.shuffle(paths)
    train_paths = paths[0:int(len(paths) * ratio[0] / sum(ratio))]
    valid_paths = paths[int(len(paths) * ratio[0] / sum(ratio)):
                               int(len(paths) * ratio[0] / sum(ratio))
                               + int(len(paths) * ratio[1] / sum(ratio))]
    test_paths = paths[int(len(paths) * ratio[0] / sum(ratio))
                              + int(len(paths) * ratio[1] / sum(ratio)):]
    return train_paths, valid_paths, test_paths



def single_domain_dataset(domain='T1', ratio=[8, 1, 1]):
    '''
    单独域的数据集划分
    '''
    paths = os.listdir('D:/study/pga/dataset/mydata2/image')
    random.seed(2023)
    # domain = 'T1'
    # ratio = [8,1,1] # 划分比例
    domain_paths = []
    for path in paths:
        if domain in path:
            domain_paths.append(path)
    random.shuffle(domain_paths)
    train_paths = domain_paths[0:int(len(domain_paths) * ratio[0] / sum(ratio))]
    valid_paths = domain_paths[int(len(domain_paths) * ratio[0] / sum(ratio)):
                               int(len(domain_paths) * ratio[0] / sum(ratio))
                               + int(len(domain_paths) * ratio[1] / sum(ratio))]
    test_paths = domain_paths[int(len(domain_paths) * ratio[0] / sum(ratio))
                              + int(len(domain_paths) * ratio[1] / sum(ratio)):]
    return train_paths, valid_paths, test_paths


def domain_generalization_dataset(domain1='T1',domain2 = 'T2', ratio=[1, 1]):
    '''
    单独域的数据集划分
    '''
    paths = os.listdir('D:/study/pga/dataset/mydata2/image')
    random.seed(2023)

    domain1_paths = []
    for path in paths:
        if domain1 in path:
            domain1_paths.append(path)

    domain2_paths = []
    for path in paths:
        if domain2 in path:
            domain2_paths.append(path)

    random.shuffle(domain2_paths)
    train_paths = domain1_paths
    valid_paths = domain2_paths[0 : int(len(domain2_paths) * ratio[0]/sum(ratio))]
    test_paths = domain2_paths[int(len(domain2_paths) * ratio[0]/sum(ratio)): ]
    return train_paths, valid_paths, test_paths

if __name__ == '__main__':
    a,b,c = single_domain_dataset('T2')
    d,e,f = domain_generalization_dataset('T1','T2')
    # paths = os.listdir('D:/study/pga/dataset/mydata2/image')
    # names = []
    # brands = []
    # types = []
    # for path in paths:
    #     tmp = path.split('_')
    #     names.append(tmp[2])
    #     brands.append(tmp[1])
    #     type = tmp[3]
    #     if 'T1' in type:
    #         types.append('T1')
    #     elif 'T2' in type:
    #         types.append('T2')
    #     elif 'DWI' in type:
    #         types.append('DWI')
    #     else:
    #         type = tmp[4]
    #         if 'T1' in type:
    #             types.append('T1')
    #         elif 'T2' in type:
    #             types.append('T2')
    #         elif 'DWI' in type:
    #             types.append('DWI')
    #         else:
    #             print('error:',path)


    # biz = 'feilipu'
    # level = 'S3'
    # person = '171_ligenfa_P'
    # MRI_type = 'DWI'
    # gold_standard = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\{MRI_type}.nii.gz'
    # predict = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\pre_{MRI_type}.nii.gz'
    # gt = sitk.GetArrayFromImage(sitk.ReadImage(gold_standard))  # ground truth
    # pre = sitk.GetArrayFromImage(sitk.ReadImage(predict))
    # print(dice_score(gt, pre, 1))
    # print(RAVD(gt,pre))
    # print(assd(gt,pre))
