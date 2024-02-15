from skimage import transform
import SimpleITK as sitk

def save_arr2nii(arr, path='tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)
    
img = sitk.ReadImage(r'D:\study\pga\dataset\mydata2\image\10_feilipu_jiabanghao_T1.nii')
img = sitk.GetArrayFromImage(img)
img2 = transform.resize(img,(32,128,128), anti_aliasing=False, preserve_range=True)
save_arr2nii(img2,'img.nii.gz')

label = sitk.ReadImage(r'D:\study\pga\dataset\mydata2\label\10_feilipu_jiabanghao_T1.nii')
label = sitk.GetArrayFromImage(label)
label2 = transform.resize(label,(32,128,128), anti_aliasing=False, preserve_range=True)
save_arr2nii(label2,'label.nii.gz')