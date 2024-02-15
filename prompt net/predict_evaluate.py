import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL.PngImagePlugin import PngImageFile

from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
from PreProcessing import DataProcessor,MyDataset # 图像预处理
from atnunet2 import AttU_Net
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标
from save_arr2nii import save_arr2nii
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

PIXEL = 256 # 这个超参数时图片resize大小，要根据模型做出改变

def predict(net, target,sid,dirname, slice_resize=(PIXEL, PIXEL)):
    '''
    给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    :param net:
    :param target:
    :return:
    '''
    img_arr = np.asarray(Image.fromarray(target).resize(slice_resize, 0))
    # 获取输入图像的尺寸
    height, width = img_arr.shape
    # 创建一个全零的三通道图像
    color_image = np.zeros((height, width, 9), dtype=np.float32)
    color_image[:, :, 0] = img_arr

    # # MRI prompt
    # if 'T1' in dirname:
    #     color_image[:, :, 1] = 1.0
    # elif 'T2' in dirname:
    #     color_image[:, :, 2] = 1.0
    # elif 'DWI' in dirname:
    #     color_image[:, :, 3] = 1.0
    #
    # # biz prompt
    # if 'feilipu' in dirname:
    #     color_image[:, :, 4] = 1.0
    # elif 'lianying' in dirname:
    #     color_image[:, :, 5] = 1.0
    # elif 'tongyong' in dirname:
    #     color_image[:, :, 6] = 1.0
    # elif 'ximenzi' in dirname:
    #     color_image[:, :, 7] = 1.0
    #
    # # sid prompt
    # color_image[:, :, 8] = sid


    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])
    img_tensor = TensorTransform(color_image)
    img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
    img_tensor4d = img_tensor4d.to(device)

    # print(type(img_tensor4d), net(img_tensor4d))
    return img_tensor4d, net(img_tensor4d)

def dir2pre(net,nii_dir,model_dir,gold_standard_dir = False,save_dir = False,standard = False,topk = 3):
    # 加载模型
    class_num = 2
    net.load_state_dict(torch.load(model_dir))  # 在此更改载入的模型
    net = net.to(device)  # 加入gpu
    # net.eval()

    lblb = sitk.ReadImage(nii_dir)
    lblb = sitk.GetArrayFromImage(lblb)
    if len(lblb.shape) == 4:  # DWI高B值
        lblb = lblb[0]
    if standard:
        lblb = (lblb - np.mean(lblb)) / np.std(lblb)  # a.先对图像做标准化
    minimum = np.min(lblb)
    gap = np.max(lblb) - minimum
    lblb = (lblb - minimum) / gap * 255  # b.再对图像做0-255“归一化”
    resize_shape = (lblb.shape[2], lblb.shape[1])
    glist = []
    for id in range(lblb.shape[0]):
        img = lblb[id].squeeze().astype(float)
        if img.shape[0] <= 5:
            sid = -1
        else:
            sid = min(id, img.shape[0] - id) / img.shape[0]  # 相对的id位置

        y_predict_arr = predict(net, img,sid,nii_dir)[1][0].squeeze(0).squeeze(0).cpu().detach().numpy() # 这个位置需要根据predict函数的修改和Unet的搭建进行适配
        img1 = y_predict_arr[1, :, :] < y_predict_arr[0, :, :]
        # img1 = y_predict_arr[0, :, :] > threshold # 修改此处更改生成label判定方式
        img1 = Image.fromarray(img1).convert('L')
        img_resize = img1.resize(resize_shape, 0)
        img_resize = np.asarray(img_resize)
        img_resize = np.expand_dims(img_resize, 0)
        glist.append(img_resize / 255)
    tmp = np.concatenate(glist, 0)
    tmp_simg = sitk.GetImageFromArray(tmp)
    # if save_dir:
    #     if 'gz' not in save_dir:
    #         sitk.WriteImage(tmp_simg, save_dir + '.gz')
    #     else:
    #         sitk.WriteImage(tmp_simg, save_dir)
    if gold_standard_dir:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gold_standard_dir))  # ground truth
        pre = sitk.GetArrayFromImage(tmp_simg)
        dice = dice_score(gt, pre, 1)
        ravd = RAVD(gt, pre)
        ans = assd(gt, pre)
        save_arr2nii(pre, 'origin.nii.gz')
        post_pre = top_k_connected_postprocessing(pre, topk)
        if save_dir:
            save_arr2nii(post_pre,save_dir)
        # 第二次后处理——填补空洞
        post_pre2 = sitk.BinaryFillhole(sitk.GetImageFromArray(post_pre))
        post_pre2 = sitk.GetArrayFromImage(post_pre2)
        if save_dir:
            save_arr2nii(post_pre2,'post_tmp.nii.gz')
        return dice,ravd,ans\
            ,dice_score(gt, post_pre, 1), RAVD(gt, post_pre), assd(gt, post_pre)\
            ,dice_score(gt, post_pre2, 1), RAVD(gt, post_pre2), assd(gt, post_pre2)



if __name__ == '__main__':
    # hyper parameter
    standard = False
    # trains, vals, tests = single_domain_dataset('T1')  # 单域数据集划分
    trains, vals, tests = cross_modality_dataset()  # 单域数据集划分
    # tests = trains[1:2]
    # trains, vals, tests = domain_generalization_dataset(domain1='lianying', domain2='tongyong', ratio=[1, 1]) # 域泛化数据集划分
    DIR = 'D:/study/pga/dataset/mydata2'  # 储存数据的绝对路径
    test1 = []
    for test in tests:
        img = f'{DIR}/image/{test}'
        label = f'{DIR}/label/{test}'
        test1.append([img,label])

    train_list,val_list,test_list = DataProcessor().get_data2([3,1,1])
    DIR = 'D:/study/pga/newtry0427/CHAOS_liver_tuning'
    for test in test_list:
        img = f'{DIR}/img/{test[3]}'
        label = f'{DIR}/label/{test[3]}'
        if [img, label] not in test1:
            test1.append([img, label])

    depth = 4
    topk = 1
    net = AttU_Net(9,2)
    model = r'best.pth'
    print(model)
    dices = []
    ravds = []
    assds = []
    dices_ = []
    ravds_ = []
    assds_ = []
    dices__ = []
    ravds__ = []
    assds__ = []
    for test in test1:
        img,label = test
        dice, ravd, ASSD\
            ,dice_, ravd_, ASSD_\
            ,dice__, ravd__, ASSD__ = dir2pre(net,img,model,label,'tmp.nii.gz',standard=False,topk=topk)
        dices.append(dice)
        ravds.append(ravd)
        assds.append(ASSD)
        dices_.append(dice_)
        ravds_.append(ravd_)
        assds_.append(ASSD_)
        dices__.append(dice__)
        ravds__.append(ravd__)
        assds__.append(ASSD__)
        print(img,dice__)
    # print(test,dice__)
    dice = np.mean(dices)
    ravd = np.mean(ravds)
    ASSD = np.mean(assds)
    dice_ = np.mean(dices_)
    ravd_ = np.mean(ravds_)
    ASSD_ = np.mean(assds_)
    dice__ = np.mean(dices__)
    ravd__ = np.mean(ravds__)
    ASSD__ = np.mean(assds__)
    print('dice:',dice)
    print('ravd:',ravd)
    print('assd:',ASSD)
    print('\n后处理结果：')
    print('dice_:', dice_)
    print('ravd_:', ravd_)
    print('assd_:', ASSD_)
    print('\n二次后处理结果：')
    print('dice__:', dice__)
    print('ravd__:', ravd__)
    print('assd__:', ASSD__)





