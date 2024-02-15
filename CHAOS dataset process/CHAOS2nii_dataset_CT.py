'''本文件的作用是把官方的CHAOS数据集加载成标准的nii文件img-label对应的数据集'''
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm


def save_arr2nii(arr, path='tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)


ids = os.listdir(r'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\CT')
for id in tqdm(ids):
    # T1 label
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\CT\{id}\Ground'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        array = np.expand_dims(array, 0)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\label_CT\{id}.nii.gz'
    save_arr2nii(img, save_dir)

    # T1 image
    dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\Train_Sets\CT\{id}\DICOM_anon'
    dir_list = sorted(os.listdir(dir))
    ans = []
    for name in dir_list:
        nii_dir = fr'{dir}\{name}'
        lblb = sitk.ReadImage(nii_dir)
        array = sitk.GetArrayFromImage(lblb)
        ans.append(array)
    img = np.concatenate(ans, 0)
    save_dir = fr'D:\study\pga\newtry0427\CHAOS_Train_Sets\img_CT\{id}.nii.gz'
    save_arr2nii(img, save_dir)




