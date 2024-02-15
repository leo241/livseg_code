'''本文件的作用是把官方的CHAOS数据集加载成标准的nii文件img-label对应的数据集'''
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm

def save_arr2nii(arr, path='tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)
    


ids = os.listdir(r'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\MR')
for id in tqdm(ids):
    # T1 label
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\MR\{id}\T1DUAL\Ground'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        array = np.expand_dims(array, 0)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\label\{id}_T1.nii.gz'
    save_arr2nii(img, save_dir)

    # T1 image
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\MR\{id}\T1DUAL\DICOM_anon\InPhase'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\img\{id}_T1.nii.gz'
    save_arr2nii(img, save_dir)

    # T2 label
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\MR\{id}\T2SPIR\Ground'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        array = np.expand_dims(array, 0)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\label\{id}_T2.nii.gz'
    save_arr2nii(img, save_dir)

    # T2 image
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\MR\{id}\T2SPIR\DICOM_anon'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\img\{id}_T2.nii.gz'
    save_arr2nii(img, save_dir)


