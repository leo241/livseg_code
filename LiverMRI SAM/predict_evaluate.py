import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from skimage import io,transform
# from torch.utils.tensorboard import SummaryWriter
# from PIL.PngImagePlugin import PngImageFile

from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
# from PreProcessing import DataProcessor,MyDataset # 图像预处理
from segment_anything import sam_model_registry
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标
from save_arr2nii import save_arr2nii

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

PIXEL = 256 # 这个超参数时图片resize大小，要根据模型做出改变


def medsam_inference(medsam_model, img_embed, box_1024=np.array([[0.0, 0.0, 1024.0, 1024.0]])):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        # boxes = None,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = low_res_pred.squeeze().cpu().detach().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def predict(net, img_np):
    '''
    给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    :param net:
    :param target:
    :return:
    '''
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np[:,:,0:3]
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    box_1024 = np.array([[0, 0, 1024, 1024]])
    with torch.no_grad(): # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
        image_embedding = net.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    medsam_seg = medsam_inference(net, image_embedding, box_1024)

    return medsam_seg

def dir2pre(net,nii_dir,gold_standard_dir = False,save_dir = False,standard = False,topk = 3):
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
        y_predict_arr = predict(net, img) # 这个位置需要根据predict函数的修改和Unet的搭建进行适配
        img1 = Image.fromarray(y_predict_arr).convert('L')
        img_resize = img1.resize(resize_shape, 0)
        img_resize = np.asarray(img_resize)
        img_resize = np.expand_dims(img_resize, 0)
        # print(np.max(img_resize))
        glist.append(img_resize)
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
    trains, vals, tests = cross_modality_dataset()  # 单域数据集划分
    # tests = tests[0:10]
    # trains, vals, tests = domain_generalization_dataset(domain1='lianying', domain2='tongyong', ratio=[1, 1]) # 域泛化数据集划分
    # trains, vals, tests = single_domain_dataset('ximenzi')  # 选择单域或者域泛化做数据集划分
    DIR = 'D:/study/pga/dataset/mydata2'  # 储存数据的绝对路径
    topk = 1
    # 载入预训练的SAM模型
    # MedSAM_CKPT_PATH = r'D:\study\pga\newtry0427\code practice\实验记录\sam_DWI_1e-5.pth'
    MedSAM_CKPT_PATH = 'best.pth'
    MedSAM_CKPT_PATH = r'D:\study\pga\newtry0427\UI design\model\sam_cross_1e-4.pth'
    net = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    net = net.to(device)
    net.eval()
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
        dice, ravd, ASSD\
            ,dice_, ravd_, ASSD_\
            ,dice__, ravd__, ASSD__ = dir2pre(net,img,label,'tmp.nii.gz',standard=False,topk=topk)
        dices.append(dice)
        ravds.append(ravd)
        assds.append(ASSD)
        dices_.append(dice_)
        ravds_.append(ravd_)
        assds_.append(ASSD_)
        dices__.append(dice__)
        ravds__.append(ravd__)
        assds__.append(ASSD__)
        print(test, dice_)
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




