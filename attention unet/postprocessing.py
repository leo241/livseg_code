import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from collections import Counter
from save_arr2nii import save_arr2nii
from evaluate_metrics import dice_score,RAVD,assd

def top_k_connected_postprocessing(mask,k):
    '''
    单连通域后处理，只保留前k个最多的连通域
    代码参考自https://blog.csdn.net/zz2230633069/article/details/85107971
    '''
    marked_label, num = label(mask, connectivity=3, return_num=True)
    shape = marked_label.shape
    lst = marked_label.flatten().tolist() # 三维数组平摊成一维列表
    freq = Counter(lst).most_common(k+1)
    keys = [item[0] for item in freq]
    for id,item in enumerate(lst):
        if item == 0:
            continue
        elif item not in keys:
            lst[id] = 0
        else:
            lst[id] = 1
    return np.asarray(lst).reshape(shape)




if __name__ == '__main__':
    topk = 1 # 保存连通域的个数
    biz = 'lianying'
    level = 'S2'
    person = '70_caofengyu'
    MRI_type = 'T1'
    gold_standard = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\{MRI_type}.nii.gz'
    predict = fr'D:\study\pga\segmentation\label\{biz}\{level}\{person}\pre_{MRI_type}.nii.gz'
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gold_standard))  # ground truth
    pre = sitk.GetArrayFromImage(sitk.ReadImage(predict))
    # marked_label,num = label(gt, connectivity=3, return_num=True)
    # print(Counter(gt.flatten().tolist()))
    # print(Counter(marked_label.flatten().tolist()))
    post_pre = top_k_connected_postprocessing(pre,topk)
    save_arr2nii(post_pre)
    print(dice_score(gt, pre, 1),RAVD(gt,pre),assd(gt,pre))
    print(dice_score(gt, post_pre, 1), RAVD(gt, post_pre), assd(gt, post_pre))

