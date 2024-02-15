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

from dataset_divide2 import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
from PreProcessing2 import DataProcessor,MyDataset # 图像预处理
# from unet import UNet # 网络结构
# from attention_unet import AttentionUNet
from BayesSeg2 import BayeSeg,BayeSeg_Criterion # 网络结构
from postprocessing2 import top_k_connected_postprocessing # mask后处理
from evaluate_metrics2 import dice_score,RAVD,assd # 评价指标
from save_arr2nii2 import save_arr2nii
import argparse
from args2 import add_management_args, add_experiment_args, add_bayes_args

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

PIXEL = 256 # 这个超参数时图片resize大小，要根据模型做出改变

def predict(net, target, slice_resize=(PIXEL, PIXEL)):
    '''
    给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    :param net:
    :param target:
    :return:
    '''
    if type(target) == str:
        img_target = Image.open(target)
        origin_size = img_target.size
        img_arr = np.asarray(img_target.resize(slice_resize, 0))
    elif type(target) == PngImageFile or type(target) == Image.Image:
        origin_size = target.size
        img_arr = np.asarray(target.resize(slice_resize, 0))
    elif type(target) == np.ndarray:
        origin_size = target.shape
        img_arr = np.asarray(Image.fromarray(target).resize(slice_resize, 0))
    else:
        print('<target type error>')
        return False
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])
    img_tensor = TensorTransform(img_arr)
    img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
    img_tensor4d = img_tensor4d.to(device)

    # print(type(img_tensor4d), net(img_tensor4d))
    return img_tensor4d, net(img_tensor4d)['pred_masks']

def dir2pre(net,nii_dir,model_dir,gold_standard_dir = False,save_dir = False,standard = False,topk = 3):
    # 加载模型
    class_num = 2
    net.load_state_dict(torch.load(model_dir))  # 在此更改载入的模型
    net = net.to(device)  # 加入gpu
    # net.eval() # 加上这行结果可能会变差很多。。。

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
        y_predict_arr = predict(net, img)[1][0].squeeze(0).squeeze(0).cpu().detach().numpy() # 这个位置需要根据predict函数的修改和Unet的搭建进行适配
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
        post_pre = top_k_connected_postprocessing(pre, topk)
        # 第二次后处理——填补空洞
        post_pre2 = sitk.BinaryFillhole(sitk.GetImageFromArray(post_pre))
        post_pre2 = sitk.GetArrayFromImage(post_pre2)
        if save_dir:
            save_arr2nii(post_pre2,save_dir)
        return dice,ravd,ans,dice_score(gt, post_pre, 1), RAVD(gt, post_pre), assd(gt, post_pre),dice_score(gt, post_pre2, 1), RAVD(gt, post_pre2), assd(gt, post_pre2)


if __name__ == '__main__':
    # hyper parameter
    standard = False
    topk = 1
    # trains, vals, tests = single_domain_dataset('feilipu')  # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = cross_modality_dataset()
    # tests = trains[0:2]
    trains, vals, tests = domain_generalization_dataset(domain1='feilipu', domain2='ximenzi', ratio=[1, 1])
    DIR = 'D:/study/pga/dataset/mydata2'  # 储存数据的绝对路径
    # batch_size = 32
    # L2 = 1e-4  # 正则化参数
    # lr = 3e-4
    # depth = 4  # Unet网络的深度
    # EPOCH = 200

    parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()
    net = BayeSeg(args)
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    criterion = BayeSeg_Criterion(args)
    # net = AttentionUNet(1,2)
    # model = r'D:\study\pga\newtry0427\code practice\实验记录\Bayes_DWI_256_0.pth'
    model = r'D:\study\pga\newtry0427\code practice\BayesSeg\final.pth'
    dices = []
    ravds = []
    assds = []
    dices_ = []
    ravds_ = []
    assds_ = []
    dices__ = []
    ravds__ = []
    assds__ = []
    for test in tests:
        img = f'{DIR}/image/{test}'
        label = f'{DIR}/label/{test}'
        dice, ravd, ASSD,dice_, ravd_, ASSD_,dice__, ravd__, ASSD__ = dir2pre(net,img,model,label,'tmp.nii.gz',standard=False,topk=topk)
        dices.append(dice)
        ravds.append(ravd)
        assds.append(ASSD)
        dices_.append(dice_)
        ravds_.append(ravd_)
        assds_.append(ASSD_)
        dices__.append(dice__)
        ravds__.append(ravd__)
        assds__.append(ASSD__)
        print(test,dice__)
    print(img)
    dice = np.mean(dices)
    ravd = np.mean(ravds)
    ASSD = np.mean(assds)
    dice_ = np.mean(dices_)
    ravd_ = np.mean(ravds_)
    ASSD_ = np.mean(assds_)
    dice__ = np.mean(dices__)
    ravd__ = np.mean(ravds__)
    ASSD__ = np.mean(assds__)
    print('dice:',dice , f'(std:{np.std(dices)})')
    print('ravd:',ravd, f'(std:{np.std(ravd)})')
    print('assd:',ASSD, f'(std:{np.std(ASSD)})')
    print('\n后处理结果：')
    print('dice_:', dice_, f'(std:{np.std(dices_)})')
    print('ravd_:', ravd_, f'(std:{np.std(ravds_)})')
    print('assd_:', ASSD_, f'(std:{np.std(assds_)})')
    print('\n二次后处理结果：')
    print('dice__:', dice__)
    print('ravd__:', ravd__)
    print('assd__:', ASSD__)


    # # predict -> use function dir2pre to
    # # 1.save predict mask or
    # # 2.give dice score between predict mask and gold standard(ground truth)
    # model_dirs = sorted(os.listdir('model_save2'))
    # model_dir1 = 'model_save_T1' # 模型储存父级路径
    # model_dirs = ['82500 - 副本.pth'] # 模型储存子路径
    # names = os.listdir('D:/study/pga/dataset/mydata2/image')
    # names_T1 = sorted([name for name in names if 'T1' in name])
    # names_T2 = sorted([name for name in names if 'T2' in name])
    # names_DWI = sorted([name for name in names if 'DWI' in name])
    # names = names_T1 + names_T2 + names_DWI
    # dic_df = {}
    # names = names[:] # 在这里对训练集样本量进行截断
    # dic_df['name'] = names
    # for model_dir in tqdm(model_dirs[:]): # 在这里对模型量进行截断
    #     dice_list = []
    #     for name in tqdm(names):
    #         # name = '31_tongyong_caomeiying_T1.nii'
    #         # model_dir = 'model_save2/200000 - 副本.pth'
    #         model_dir_whole = model_dir1 + '/' + model_dir
    #         nii_dir = f'D:/study/pga/dataset/mydata2/image/{name}'
    #         gold_standard_dir = f'D:/study/pga/dataset/mydata2/label/{name}'
    #         dice = dir2pre(nii_dir,model_dir_whole,gold_standard_dir = gold_standard_dir, save_dir=False, standard=standard)
    #         dice_list.append(dice)
    #     dic_df[model_dir] = dice_list
    # df = pd.DataFrame(dic_df)
    # df.to_csv('train_evaluate/T1_0428_2046.csv',index=False)




