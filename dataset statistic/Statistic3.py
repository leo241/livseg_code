'''本文件的作用为统计厂家模态下的病人数量'''
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
from evaluate_metrics import dice_dir1_dir2

dir_list = os.listdir(r'D:\study\pga\dataset\mydata2\image')
feilipu = []
lianying = []
ximenzi = []
tongyong = []
for name in dir_list:
    if 'feilipu' in name:
        feilipu.append(name.split('_')[2])
    elif 'lianying' in name:
        lianying.append(name.split('_')[2])
    elif 'ximenzi' in name:
        ximenzi.append(name.split('_')[2])
    elif 'tongyong' in name:
        tongyong.append(name.split('_')[2])
feilipu2 = list(set(feilipu))
lianying2 = list(set(lianying))
ximenzi2 = list(set(ximenzi))
tongyong2 = list(set(tongyong))

dirlist2 = sorted(dir_list,key=lambda x:x.split('_')[2])