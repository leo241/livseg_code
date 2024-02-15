import numpy as np
import SimpleITK as sitk
import torch

from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
from model import UNet3D # 网络结构
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标

def save_arr2nii(arr,path = 'tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)
from tqdm import tqdm

from skimage import transform

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速


def resize_3d_image(image, size = (32,120,120)):
    image =transform.resize(image,size, anti_aliasing=False, preserve_range=True) # 这两个参数的设置非常重要，抗锯齿，且在插值时保持数据值的大小
    return image

def predict(net, target, slice_resize=(32,120,120)):
    '''
    给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    :param net:
    :param target:
    :return:
    '''

    img_arr = resize_3d_image(target)
    img_tensor = torch.from_numpy(img_arr)
    img_tensor4d = img_tensor.unsqueeze(0).unsqueeze(0)  # 只有把图像3维（32，120，120）扩展成4维（1，1，32，120，120）才能放进神经网络预测
    img_tensor4d = img_tensor4d.to(device)
    img_tensor4d = img_tensor4d.to(torch.float)
    return img_tensor4d, net(img_tensor4d)

def dir2pre(net,nii_dir,model_dir,gold_standard_dir = False,save_dir = False,standard = False,topk = 3):
    # 加载模型
    net.load_state_dict(torch.load(model_dir))  # 在此更改载入的模型
    net = net.to(device)  # 加入gpu
    # net.eval()

    lblb = sitk.ReadImage(nii_dir)
    lblb = sitk.GetArrayFromImage(lblb)
    lblb = lblb.reshape(tuple(sorted(lblb.shape)))  # 从小到大顺序
    if len(lblb.shape) == 4:  # DWI高B值
        lblb = lblb[0]
    if standard:
        lblb = (lblb - np.mean(lblb)) / np.std(lblb)  # a.先对图像做标准化
    minimum = np.min(lblb)
    gap = np.max(lblb) - minimum
    lblb = (lblb - minimum) / gap * 255  # b.再对图像做0-255“归一化”
    resize_shape = (lblb.shape[0], lblb.shape[1],lblb.shape[2])

    # img = lblb.astype(float)
    img = lblb
    y_predict_arr = predict(net, img)[1][0].squeeze(0).squeeze(0).cpu().detach().numpy() # 这个位置需要根据predict函数的修改和Unet的搭建进行适配
    img1 = np.zeros(y_predict_arr.shape, dtype=np.uint8)
    img1[(y_predict_arr >= 0.5)] = 1

    img_resize = resize_3d_image(img1,size = resize_shape)
    tmp_simg = sitk.GetImageFromArray(img_resize)
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
        tmp = post_pre2
        post_pre2 = sitk.GetArrayFromImage(post_pre2)
        if save_dir:
            save_arr2nii(post_pre2,'post_tmp.nii.gz')
        # 第三次后处理——形态学闭运算
        # 定义结构元素（这里是一个3x3x3的立方体结构元素）
        # structuring_element = sitk.BinaryBallStructuringElement(3, sitk.sitkBox)

        # 为了将内部空洞完全填充，通常会先侵蚀后膨胀，这个组合被称为“闭运算”（Closing）
        post_pre3 = sitk.BinaryMorphologicalClosing(tmp)

        # sitk.BinaryMorphologicalClosing(sitk.ReadImage() != 0, kernelsize)  # 闭
        post_pre3 = sitk.GetArrayFromImage(post_pre3)
        if save_dir:
            save_arr2nii(post_pre3,'post_tmp3.nii.gz')
        return dice,ravd,ans\
            ,dice_score(gt, post_pre, 1), RAVD(gt, post_pre), assd(gt, post_pre)\
            ,dice_score(gt, post_pre2, 1), RAVD(gt, post_pre2), assd(gt, post_pre2),dice_score(gt, post_pre3, 1), RAVD(gt, post_pre3), assd(gt, post_pre3)





if __name__ == '__main__':
    # hyper parameter
    standard = False
    # trains, vals, tests = single_domain_dataset('T2')  # 单域数据集划分
    # trains, vals, tests = cross_modality_dataset()  # 单域数据集划分
    # tests = trains[1:2]
    trains, vals, tests = domain_generalization_dataset(domain1='lianying', domain2='tongyong', ratio=[1, 1]) # 域泛化数据集划分
    DIR = 'D:/study/pga/dataset/mydata2'  # 储存数据的绝对路径
    topk = 1
    net =  UNet3D(in_channels=1,out_channels=1)
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
    dices___ = []
    ravds___ = []
    assds___ = []
    for test in tests:
        img = f'{DIR}/image/{test}'
        label = f'{DIR}/label/{test}'
        dice, ravd, ASSD\
            ,dice_, ravd_, ASSD_\
            ,dice__, ravd__, ASSD__,dice___, ravd___, ASSD___ = dir2pre(net,img,model,label,'tmp.nii.gz',standard=False,topk=topk)
        dices.append(dice)
        ravds.append(ravd)
        assds.append(ASSD)
        dices_.append(dice_)
        ravds_.append(ravd_)
        assds_.append(ASSD_)
        dices__.append(dice__)
        ravds__.append(ravd__)
        assds__.append(ASSD__)
        dices___.append(dice___)
        ravds___.append(ravd___)
        assds___.append(ASSD___)
        print(test,dice___)
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
    dice___ = np.mean(dices___)
    ravd___ = np.mean(ravds___)
    ASSD___ = np.mean(assds___)
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
    print('\n三次后处理结果：')
    print('dice___:', dice___)
    print('ravd___:', ravd___)
    print('assd___:', ASSD___)


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




