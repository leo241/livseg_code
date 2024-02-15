def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script

import os.path

from PyQt5.QtWidgets import QApplication, QWidget\
    , QDesktopWidget, QMessageBox, QToolTip, QLabel\
    , QTabWidget,QLineEdit,QPushButton, QComboBox,QProgressBar,QFileDialog,QSplashScreen,QMainWindow,QVBoxLayout
from PyQt5.QtGui import QPalette, QColor, QIcon, QFont, QPainter,QPen, QPixmap,QCursor,QBrush # 调色板
from PyQt5.QtCore import Qt,QRect,pyqtSignal,QThread,QSize
# from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import qtawesome
# import pandas as pd
# import joblib
import argparse
import warnings
import SimpleITK as sitk
from model.unet import UNet
from model.atnunet2 import AttU_Net
from model.deeplab.deeplabv3 import DeepLab
from model.bayeseg.BayesSeg2 import BayeSeg,BayeSeg_Criterion
from model.bayeseg.args2 import add_management_args, add_experiment_args, add_bayes_args
from model.segment_anything import sam_model_registry
from model.unet3d.model import UNet3D
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from PIL.PngImagePlugin import PngImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform # 用于sam的推断过程
from tqdm import tqdm
from skimage.measure import label
from collections import Counter
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速


def save_arr2nii(arr, path='tmp.nii.gz'):
    tmp_simg = sitk.GetImageFromArray(arr)
    sitk.WriteImage(tmp_simg, path)

def resize_3d_image(image, size = (32,120,120)):
    image =transform.resize(image,size, anti_aliasing=False, preserve_range=True) # 这两个参数的设置非常重要，抗锯齿，且在插值时保持数据值的大小
    return image

def convert_to_color(image): # 将黑白图片变成黑白的彩色图片
    # 获取输入图像的尺寸
    height, width = image.shape

    # 创建一个全零的三通道图像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将输入图像复制到每个通道中
    for i in range(3):
        color_image[:, :, i] = image

    return color_image

def convert_to_red(image): # 用于涂红标签
    # 获取输入图像的尺寸
    height, width = image.shape

    # 创建一个全零的三通道图像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将输入图像复制到每个通道中

    color_image[:, :, 0] = image

    return color_image


# 继承QThread
class Predictle_thread(QThread): # 线程QThread里面最关键的是run函数
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str)
    _signal2 = pyqtSignal(str)

    def __init__(self,file_names,model_name,topk_info,hole_fill_info,if_save_info,save_path):
        super(Predictle_thread, self).__init__()
        self.file_names = file_names
        self.model_name = model_name # combo box中的模型选项
        self.top_k_info = topk_info
        self.hole_fill_info = hole_fill_info
        self.if_save_info = if_save_info
        self.save_path = save_path
        self.img = None

    def __del__(self):
        self.wait()

    def get_model(self,model_name): # 返回网络结构和模型名称
        if model_name == 'Unet':
            net = UNet(num_classes=2, depth=4, size=256, rgb_channel=1, initial_channel=8) # 更改网络
            # model = 'model/unet4_T1_0.pth' # T1专属模型
            model = 'model/unet.pth' # 全模态
            slice_resize = (256,256)
            topk = 1 # 先写死，后面变为灵活调整
            return net,model,slice_resize,topk
        elif model_name == 'Attenton Unet':
            net = AttU_Net(1,2)
            model = 'model/atnunet.pth'
            slice_resize = (256, 256)
            topk = 1 # 先写死，后面变为灵活调整
            return net, model, slice_resize, topk
        elif model_name == 'DeepLab': # 输入须改为rbg三通道
            net = DeepLab(num_classes=2,pretrained=False)
            model = 'model/deeplab.pth'
            slice_resize = (256, 256)
            topk = 1 # 先写死，后面变为灵活调整
            return net, model, slice_resize, topk
        elif model_name == 'unet3d': # 输入须改为rbg三通道
            net = UNet3D(in_channels=1,out_channels=1)
            model = 'model/unet3d.pth'
            slice_resize = (32,120,120)
            topk = 1 # 先写死，后面变为灵活调整
            return net, model, slice_resize, topk
        elif model_name == 'BayeSeg': # 预测输出以字典形式在predict_mask里面
            parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
            add_experiment_args(parser)
            add_management_args(parser)
            add_bayes_args(parser)
            args = parser.parse_args()
            net = BayeSeg(args)
            model = 'model/bayeseg.pth'
            slice_resize = (256, 256)
            topk = 1 # 先写死，后面变为灵活调整
            return net, model, slice_resize, topk
        elif model_name == 'MRI_SAM': # 输入须改为rbg三通道
            model = 'model/sam.pth'
            net = sam_model_registry['vit_b'](checkpoint=model)
            slice_resize = (256, 256)
            topk = 1 # 先写死，后面变为灵活调整
            return net, model, slice_resize, topk
        elif model_name == 'ensemble':
            net1 = UNet(num_classes=2, depth=4, size=256, rgb_channel=1, initial_channel=8)  # 更改网络
            net2 = AttU_Net(1, 2)
            net3 = DeepLab(num_classes=2, pretrained=False)
            net4 = sam_model_registry['vit_b'](checkpoint='model/sam.pth')
            net4.eval()
            net5 = UNet3D(in_channels=1,out_channels=1)

            net1.load_state_dict(torch.load('model/unet.pth', map_location=device))  # 在此更改载入的模型
            net2.load_state_dict(torch.load('model/atnunet.pth', map_location=device))  # 在此更改载入的模型
            net3.load_state_dict(torch.load('model/deeplab.pth', map_location=device))
            net5.load_state_dict(torch.load('model/unet3d.pth', map_location=device))

            net1 = net1.to(device)
            net2 = net2.to(device)
            net3 = net3.to(device)
            net4 = net4.to(device)
            net5 = net5.to(device)
            return [net1,net2,net3,net4,net5],'ensemble',(256, 256),1

    def medsam_inference(self,medsam_model, img_embed, box_1024=np.array([[0.0, 0.0, 1024.0, 1024.0]])):  # sam的推断函数
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

    def predict(self,net, target, slice_resize=(256, 256)):
        '''
        给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
        :param net:
        :param target:
        :return:
        '''
        if self.model_name == 'ensemble':
            img_arr = np.asarray(Image.fromarray(target).resize(slice_resize, 0))
            TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
                transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
            ])
            img_tensor = TensorTransform(img_arr)
            img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
            img_tensor4d = img_tensor4d.to(device)
            ans1 = net[0](img_tensor4d)[0].squeeze(0).squeeze(
                            0).cpu().detach().numpy()
            ans1 = ans1[1, :, :] < ans1[0, :, :]
            ans2 = net[1](img_tensor4d)[0].squeeze(0).squeeze(
                            0).cpu().detach().numpy()
            ans2 = ans2[1, :, :] < ans2[0, :, :]
            img_arr2 = convert_to_color(img_arr) # deeplab
            img_tensor = TensorTransform(img_arr2)
            img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
            img_tensor4d = img_tensor4d.to(device)
            ans3 = net[2](img_tensor4d)[0].squeeze(0).squeeze(
                            0).cpu().detach().numpy()
            ans3 = ans3[1, :, :] < ans3[0, :, :]



            img_np = target
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np[:, :, 0:3]
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
            with torch.no_grad():  # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
                image_embedding = net[3].image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
            ans4 = self.medsam_inference(net[3], image_embedding, box_1024)

            ensemble = [ans1,ans2,ans3,ans4,self.ans5]
            ensemble = sum(ensemble)/len(ensemble)
            label_ensemble = np.zeros(ensemble.shape)
            label_ensemble[(ensemble >= 0.5)] = 1
            return 0,label_ensemble

        if self.model_name == 'MRI_SAM': # sam的predict推断函数比较特别，单独写出
            img_np = target
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np[:, :, 0:3]
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
            with torch.no_grad():  # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
                image_embedding = net.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
            medsam_seg = self.medsam_inference(net, image_embedding, box_1024)

            return image_embedding,medsam_seg

        if self.model_name == 'unet3d':
            img_arr = resize_3d_image(target)
            img_tensor = torch.from_numpy(img_arr)
            img_tensor4d = img_tensor.unsqueeze(0).unsqueeze(0)  # 只有把图像3维（32，120，120）扩展成4维（1，1，32，120，120）才能放进神经网络预测
            img_tensor4d = img_tensor4d.to(device)
            img_tensor4d = img_tensor4d.to(torch.float)
            return img_tensor4d, net(img_tensor4d)

        try:
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
            if self.model_name == 'DeepLab': # Deeplab需要输入的图片是rgb图片
                img_arr = convert_to_color(img_arr)
            img_tensor = TensorTransform(img_arr)
            img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
            img_tensor4d = img_tensor4d.to(device)
            ans = net(img_tensor4d)
        except Exception as predict_er:
            print('内部预测错误',predict_er)
            raise Exception('predict fail!')

        # print(type(img_tensor4d), net(img_tensor4d))
        return img_tensor4d, ans

    def top_k_connected_postprocessing(self,mask, k):
        '''
        单连通域后处理，只保留前k个最多的连通域
        代码参考自https://blog.csdn.net/zz2230633069/article/details/85107971
        '''
        marked_label, num = label(mask, connectivity=3, return_num=True)
        shape = marked_label.shape
        lst = marked_label.flatten().tolist()  # 三维数组平摊成一维列表
        freq = Counter(lst).most_common(k + 1)
        keys = [item[0] for item in freq]
        for id, item in enumerate(lst):
            if item == 0:
                continue
            elif item not in keys:
                lst[id] = 0
            else:
                lst[id] = 1
        return np.asarray(lst).reshape(shape)

    def save_arr2nii(self,arr, path='tmp.nii.gz'):
        if os.path.splitext(path)[-1] in ['.jpg','.JPG','.jpeg','.JPEG']: # jpg文件无法三维写入
            arr = arr[0]
        tmp_simg = sitk.GetImageFromArray(arr)
        tmp_simg = sitk.Cast(tmp_simg, sitk.sitkUInt8) # 做完这个格式转化，就可以以png的格式存储标签了，实现了各种图像都能储存
        sitk.WriteImage(tmp_simg, path)

    def dir2pre(self,net, nii_dir, model_dir,root_save_path, save_dir=False, standard=False, topk=3,slice_size = (256,256)):
        # 加载模型
        if model_dir != 'ensemble':
            if self.model_name != 'MRI_SAM':
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))  # 在此更改载入的模型
                    print('cpu loading')
                else:
                    net.load_state_dict(torch.load(model_dir))  # 在此更改载入的模型
            net = net.to(device)  # 加入gpu
            if self.model_name == 'MRI_SAM':
                net.eval()
        # net.eval() # 加上这行结果可能会变差很多

        for item in nii_dir:
            self._signal2.emit(item)
            lblb = sitk.ReadImage(item)
            # print('fig dir:',item)
            lblb = sitk.GetArrayFromImage(lblb)
            # print('处理前shape', lblb.shape)
            lblb = lblb.reshape(tuple(sorted(lblb.shape))) # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
            if len(lblb.shape) == 2:  # 二维图片需要扩充维度
                lblb = lblb.reshape((1,lblb.shape[0],lblb.shape[1]))
            if len(lblb.shape) == 4:  # DWI高B值图像是四维，直接忽略最短的维度
                lblb = lblb[0]
            if standard:
                lblb = (lblb - np.mean(lblb)) / np.std(lblb)  # a.先对图像做标准化
            # print('处理后shape', lblb.shape)
            minimum = np.min(lblb)
            gap = np.max(lblb) - minimum
            lblb = (lblb - minimum) / gap * 255  # b.再对图像做0-255“归一化”
            if self.model_name == 'unet3d':
                try:
                    resize_shape = (lblb.shape[0], lblb.shape[1], lblb.shape[2])
                    y_predict_arr = self.predict(net, lblb, slice_resize=slice_size)[1][0].squeeze(0).squeeze(0).cpu().detach().numpy()
                    img1 = np.zeros(y_predict_arr.shape, dtype=np.uint8)
                    img1[(y_predict_arr >= 0.5)] = 1

                    tmp = resize_3d_image(img1, size=resize_shape)
                    post_pre = self.top_k_connected_postprocessing(tmp, k=int(self.top_k_info))  # 后处理
                    post_pre2 = sitk.BinaryFillhole(sitk.GetImageFromArray(post_pre)) # 孔洞填充
                    post_pre2 = sitk.BinaryMorphologicalClosing(post_pre2)
                    post_pre2 = sitk.GetArrayFromImage(post_pre2) # Unet3d模型必须进行孔洞填充和形态学实心
                    if save_dir:
                        save_path = os.path.basename(item)
                        save_path_lst = save_path.split('.')
                        save_path_lst[0] = save_path_lst[0] + '_label'
                        final_save_dir = os.path.join(root_save_path, '.'.join(save_path_lst))  # 根保存文件夹和图像名的路径拼接
                        self.save_arr2nii(post_pre2, final_save_dir)
                    img = lblb
                except Exception as er_3d:
                    print('unet3d err:', er_3d)
                    print('报错行数：', er_3d.__traceback__.tb_lineno)
                continue # unet3d 流程自己走完
            if self.model_name == 'ensemble': # 在二维predict之前，unet3d先自己操作一下
                resize_shape = (lblb.shape[0], lblb.shape[1], lblb.shape[2])
                self.model_name = 'unet3d'
                y_predict_arr = self.predict(net[4], lblb, slice_resize=slice_size)[1][0].squeeze(0).squeeze(0).cpu().detach().numpy()
                self.model_name = 'ensemble'
                img1 = np.zeros(y_predict_arr.shape, dtype=np.uint8)
                img1[(y_predict_arr >= 0.5)] = 1

                tmp = resize_3d_image(img1, size=resize_shape)
                post_pre = self.top_k_connected_postprocessing(tmp, k=int(self.top_k_info))  # 后处理
                post_pre2 = sitk.BinaryFillhole(sitk.GetImageFromArray(post_pre)) # 孔洞填充
                post_pre2 = sitk.BinaryMorphologicalClosing(post_pre2)
                post_pre2 = sitk.GetArrayFromImage(post_pre2) # Unet3d模型必须进行孔洞填充和形态学实心
                self.mask3d = resize_3d_image(post_pre2,size=(lblb.shape[0],256,256)) # 为后面集成做准备


            resize_shape = (lblb.shape[2], lblb.shape[1])
            glist = []
            zlist = []
            for id in range(lblb.shape[0]):
                self._signal.emit(str(int(id / lblb.shape[0] * 100)))  # 向进度条报告现在的进度
                img = lblb[id].squeeze().astype(float)
                try:
                    if self.model_name == 'ensemble':
                        self.ans5 = self.mask3d[id]
                    ans = self.predict(net, img,slice_resize=slice_size)[1]
                    if self.model_name == 'BayeSeg': # Bayes的代码层级预测结果以字典形式
                        ans = ans['pred_masks']
                    if self.model_name != 'MRI_SAM' and self.model_name != 'ensemble':
                        y_predict_arr = ans[0].squeeze(0).squeeze(
                            0).cpu().detach().numpy()  # 这个位置需要根据predict函数的修改和Unet的搭建进行适配
                        img1 = y_predict_arr[1, :, :] < y_predict_arr[0, :, :]
                    else: # sam的推断直接得到mask的np
                        img1 = ans * 255
                except Exception as predict_error:
                    print(predict_error)
                    raise Exception("Predict fail!")

                # img1 = y_predict_arr[0, :, :] > threshold # 修改此处更改生成label判定方式
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(resize_shape, 0)
                img_resize = np.asarray(img_resize)
                img_resize = np.expand_dims(img_resize, 0)
                z_resize = np.expand_dims(img, 0)
                glist.append(img_resize / 255)
                zlist.append(z_resize)
            tmp = np.concatenate(glist, 0)
            img = np.concatenate(zlist, 0)
            if self.top_k_info !='max': # 根据选项做topk后处理
                post_pre = self.top_k_connected_postprocessing(tmp, k = int(self.top_k_info)) # 后处理
            else:
                post_pre = tmp.astype(np.uint8)
            # 第二次后处理——填补空洞
            if self.hole_fill_info == '开':
                post_pre2 = sitk.BinaryFillhole(sitk.GetImageFromArray(post_pre))
                post_pre2 = sitk.GetArrayFromImage(post_pre2)
            elif self.hole_fill_info == '关':
                post_pre2 = post_pre
            else:
                raise Exception('二次后处理选项异常')
            if save_dir:
                save_path = os.path.basename(item)
                save_path_lst = save_path.split('.')
                save_path_lst[0] = save_path_lst[0] + '_label'
                final_save_dir = os.path.join(root_save_path,'.'.join(save_path_lst)) # 根保存文件夹和图像名的路径拼接
                self.save_arr2nii(post_pre2, final_save_dir)
        return img,post_pre2


    def run(self):
        net,model,slice_resize,topk = self.get_model(self.model_name)
        try:
            if self.if_save_info == '保存':
                save_dir = True
            else:
                save_dir = False
            self.img,self.post_mask = self.dir2pre(net, self.file_names, model,root_save_path=self.save_path, save_dir=save_dir, standard=False, topk=topk,slice_size=slice_resize)
        except Exception as er:
            print('run 函数error',er)
            print('报错行数：', er.__traceback__.tb_lineno)
        self._signal.emit(str(100)) # 进度条完成的标志
        # print('线程结束！')
        # for i in range(100):
        #     print(i)
        #     time.sleep(0.05)
        #     self._signal.emit(str(i))  # 注意这里与_signal = pyqtSignal(str)中的类型相同
        # self._signal.emit(str(100))

class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data):
        self.data = data
        self.TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, mask, id = self.data[item]
        img_arr = np.asarray(img)
        img_arr = np.expand_dims(img_arr, 2)  # (512，512)->(512，512,1) # 实际图像矩阵

        return self.TensorTransform(img_arr), self.TensorTransform(mask), torch.tensor(id)

    def __len__(self):
        return len(self.data)

class MyDataset_deeplab(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data):
        self.data = data
        self.TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, mask, id = self.data[item]
        img_arr = np.asarray(img)
        # img_arr = np.expand_dims(img_arr, 2)  # (512，512)->(512，512,1) # 实际图像矩阵
        img_arr = convert_to_color(img_arr)

        return self.TensorTransform(img_arr), self.TensorTransform(mask), torch.tensor(id)

    def __len__(self):
        return len(self.data)

class MyDataset_sam(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data):
        self.data = data
        self.TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img_np, label1, id = self.data[item]
        # img
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1)
        )
        # label
        label1 = Image.fromarray(label1).convert('L')
        label_resize = label1.resize((256, 256), 0)
        label_resize = np.asarray(label_resize)
        label_resize = np.expand_dims(label_resize, 0)
        label_resize = torch.tensor(label_resize).to(torch.float)

        return img_1024_tensor, label_resize, torch.tensor(id)

    def __len__(self):
        return len(self.data)

class tuning_thread(QThread): # 线程QThread里面最关键的是run函数
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str)

    def __init__(self,dir_img,dir_label,model_name,epoch):
        super(tuning_thread, self).__init__()
        self.dir_img = dir_img
        self.dir_label = dir_label
        self.model_name = model_name # combo box中的模型选项
        self.epoch = epoch
        self.slice_resize = (256,256) # 将所有切片先resize成256再做训练微调，这个超参数固定写死了

    def __del__(self):
        self.wait()

    def mask_one_hot(self,
                     img_arr):  # 将label（512，512）转化为标准的mask形式（512，512，class_num）
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1), 注意这里传进来的不是img，而是label
        mask_shape = img_arr.shape
        mask1 = np.zeros(mask_shape)
        mask2 = np.zeros(mask_shape)
        mask1[img_arr > 0] = 1  # foreground
        mask2[img_arr == 0] = 1
        mask = np.concatenate([mask1, mask2], 2)
        return mask

    def medsam_inference(self,medsam_model, img_embed, box_1024 = np.array([[0.0, 0.0, 1024.0, 1024.0]])):
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
        return low_res_pred

    def run(self):
        try:
            model_name = self.model_name
            epochs = self.epoch
            dir_img = self.dir_img
            dir_label = self.dir_label
            samples = sorted(os.listdir(dir_img))
            if model_name == 'Unet':
                net = UNet(num_classes=2, depth=4, size=256, rgb_channel=1, initial_channel=8)  # 更改网络
                model = 'model/unet.pth'  # 全模态
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))  # 在此更改载入的模型
                    print('cpu loading')
                else:
                    net.load_state_dict(torch.load(model))  # 在此更改载入的模型
                net = net.to(device)  # 加入gpu
                net.train()
                for epoch in range(epochs):
                    for sample in samples: # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch+1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img,sample))
                            label = sitk.ReadImage(os.path.join(dir_label,sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。',er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))# 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))# 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        train_list = []
                        for id in range(img.shape[0]): # 把一个图像切片做预处理
                            img1 = img[id, :, :]
                            label1 = label[id, :, :]
                            img1 = Image.fromarray(img1).convert('L')
                            img_resize = img1.resize(self.slice_resize, 0)
                            label1 = Image.fromarray(label1).convert('L')
                            label_resize = label1.resize(self.slice_resize, 0)
                            label_resize = self.mask_one_hot(label_resize)
                            train_list.append([img_resize, label_resize, id])
                        train_data = MyDataset(train_list) # 一个train data只放一组图像
                        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的

                        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4) # 超参数写死
                        for step, (x, y, y2) in enumerate(train_loader):
                            x, y, y2 = x.to(device), y.to(device), y2.to(device)
                            output1, output2 = net(x)
                            output1 = output1.to(torch.float)
                            y = y.to(torch.float)
                            output2 = output2.to(torch.float)
                            y2 = y2.to(torch.float)
                            loss1 = nn.BCEWithLogitsLoss()(output1, y)
                            loss = loss1 + nn.MSELoss()(output2, y2).to(torch.float) * 0.01
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(net.state_dict(), model) # 每喂完一个epoch就直接更新参数
            elif model_name == 'Attenton Unet':
                net = AttU_Net(1, 2)
                model = 'model/atnunet.pth'
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))  # 在此更改载入的模型
                    print('cpu loading')
                else:
                    net.load_state_dict(torch.load(model))  # 在此更改载入的模型
                net = net.to(device)  # 加入gpu
                net.train()
                for epoch in range(epochs):
                    for sample in samples:  # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch + 1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img, sample))
                            label = sitk.ReadImage(os.path.join(dir_label, sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。', er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        train_list = []
                        for id in range(img.shape[0]):  # 把一个图像切片做预处理
                            img1 = img[id, :, :]
                            label1 = label[id, :, :]
                            img1 = Image.fromarray(img1).convert('L')
                            img_resize = img1.resize(self.slice_resize, 0)
                            label1 = Image.fromarray(label1).convert('L')
                            label_resize = label1.resize(self.slice_resize, 0)
                            label_resize = self.mask_one_hot(label_resize)
                            train_list.append([img_resize, label_resize, id])
                        train_data = MyDataset(train_list)  # 一个train data只放一组图像
                        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的

                        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)  # 超参数写死
                        for step, (x, y, y2) in enumerate(train_loader):
                            x, y, y2 = x.to(device), y.to(device), y2.to(device)
                            output1 = net(x)
                            output1 = output1.to(torch.float)
                            y = y.to(torch.float)
                            loss1 = nn.BCEWithLogitsLoss()(output1, y)
                            loss = loss1
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(net.state_dict(), model)  # 每喂完一个epoch就直接更新参数
            elif model_name == 'unet3d':
                net = UNet3D(in_channels=1,out_channels=1)
                model = 'model/unet3d.pth'
                net.load_state_dict(torch.load(model, map_location=device))  # 在此更改载入的模型
                net = net.to(device)  # 加入gpu
                net.train()
                for epoch in range(epochs):
                    for sample in samples:  # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch + 1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img, sample))
                            label = sitk.ReadImage(os.path.join(dir_label, sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。', er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        img = resize_3d_image(img)  # resize
                        label = resize_3d_image(label)  # resize
                        x = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0),0)) # img tensor
                        y = torch.from_numpy(np.expand_dims(np.expand_dims(label, 0),0)) # label tensor
                        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)  # 超参数写死
                        x, y = x.to(device), y.to(device)
                        x = x.to(torch.float)
                        output1 = net(x)
                        output1 = output1.to(torch.float)
                        y = y.to(torch.float)
                        loss = nn.MSELoss()(output1, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    torch.save(net.state_dict(), model)  # 每喂完一个epoch就直接更新参数
            elif model_name == 'DeepLab':  # 输入须改为rbg三通道
                net = DeepLab(num_classes=2, pretrained=False)
                model = 'model/deeplab.pth'
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))  # 在此更改载入的模型
                    print('cpu loading')
                else:
                    net.load_state_dict(torch.load(model))  # 在此更改载入的模型
                net = net.to(device)  # 加入gpu
                net.train()
                for epoch in range(epochs):
                    for sample in samples:  # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch + 1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img, sample))
                            label = sitk.ReadImage(os.path.join(dir_label, sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。', er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        train_list = []
                        for id in range(img.shape[0]):  # 把一个图像切片做预处理
                            img1 = img[id, :, :]
                            label1 = label[id, :, :]
                            img1 = Image.fromarray(img1).convert('L')
                            img_resize = img1.resize(self.slice_resize, 0)
                            label1 = Image.fromarray(label1).convert('L')
                            label_resize = label1.resize(self.slice_resize, 0)
                            label_resize = self.mask_one_hot(label_resize)
                            train_list.append([img_resize, label_resize, id])
                        train_data = MyDataset_deeplab(train_list)  # 一个train data只放一组图像,这里要求deeplab的输入是rgb图像，特殊处理
                        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的

                        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)  # 超参数写死
                        for step, (x, y, y2) in enumerate(train_loader):
                            x, y, y2 = x.to(device), y.to(device), y2.to(device)
                            output1 = net(x)
                            output1 = output1.to(torch.float)
                            y = y.to(torch.float)
                            loss1 = nn.BCEWithLogitsLoss()(output1, y)
                            loss = loss1
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(net.state_dict(), model)  # 每喂完一个epoch就直接更新参数
            elif model_name == 'BayeSeg':  # 预测输出以字典形式在predict_mask里面
                parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
                add_experiment_args(parser)
                add_management_args(parser)
                add_bayes_args(parser)
                args = parser.parse_args()
                args.bayes_loss_coef = 0  # 更改贝叶斯损失的权重
                net = BayeSeg(args)
                model = 'model/bayeseg.pth'
                criterion = BayeSeg_Criterion(args)
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))  # 在此更改载入的模型
                    print('cpu loading')
                else:
                    net.load_state_dict(torch.load(model))  # 在此更改载入的模型
                net = net.to(device)  # 加入gpu
                for epoch in range(epochs):
                    for sample in samples:  # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch + 1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img, sample))
                            label = sitk.ReadImage(os.path.join(dir_label, sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。', er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        train_list = []
                        for id in range(img.shape[0]):  # 把一个图像切片做预处理
                            img1 = img[id, :, :]
                            label1 = label[id, :, :]
                            img1 = Image.fromarray(img1).convert('L')
                            img_resize = img1.resize(self.slice_resize, 0)
                            label1 = Image.fromarray(label1).convert('L')
                            label_resize = label1.resize(self.slice_resize, 0)
                            label_resize = self.mask_one_hot(label_resize)
                            train_list.append([img_resize, label_resize, id])
                        train_data = MyDataset(train_list)  # 一个train data只放一组图像
                        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的

                        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)  # 超参数写死
                        for step, (x, y, y2) in enumerate(train_loader):
                            x, y, y2 = x.to(device), y.to(device), y2.to(device)
                            output1 = net(x)
                            loss, loss_dict = criterion(output1, y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(net.state_dict(), model)  # 每喂完一个epoch就直接更新参数
            elif model_name == 'MRI_SAM':  # 输入须改为rbg三通道
                model = 'model/sam.pth'
                net = sam_model_registry['vit_b'](checkpoint=model)
                net = net.to(device)
                net.train()
                for epoch in range(epochs):
                    for sample in samples:  # 我的想法是内存里每次只进一组图像，以防止爆掉内存
                        self._signal.emit(f'tuning info - epoch ({epoch + 1}) image ({sample})')
                        try:
                            img = sitk.ReadImage(os.path.join(dir_img, sample))
                            label = sitk.ReadImage(os.path.join(dir_label, sample))
                        except Exception as er_readimg:
                            print('读入{sample}图像和标签失败，请检验文件类型。', er_readimg)
                        img = sitk.GetArrayFromImage(img)
                        label = sitk.GetArrayFromImage(label)
                        img = img.reshape(tuple(sorted(img.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        label = label.reshape(tuple(sorted(label.shape)))  # 让维度按照从小到大的顺序重组,我建议训练的时候也作同样的数据增强
                        if img.shape != label.shape:
                            print(f'{sample}图像与标签形状不一致！')
                            continue
                        if len(img.shape) > 3:
                            print(f'{sample}图像在三维以上，无法处理！')
                            continue
                        if len(img.shape) == 2:  # 二维图片需要扩充维度
                            img = img.reshape((1, img.shape[0], img.shape[1]))
                            label = label.reshape((1, label.shape[0], label.shape[1]))
                        minimum = np.min(img)
                        gap = np.max(img) - minimum
                        img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
                        train_list = []
                        for id in range(img.shape[0]):  # 把一个图像切片做预处理
                            img1 = img[id, :, :]
                            label1 = label[id, :, :]
                            train_list.append([img1, label1, id])
                        train_data = MyDataset_sam(train_list)  # 一个train data只放一组图像,这里要求deeplab的输入是rgb图像，特殊处理
                        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的

                        optimizer = torch.optim.Adam(net.mask_decoder.parameters(), lr=3e-5, weight_decay=1e-4)  # 超参数写死,只对sam decoder部分微调
                        for step, (x, y, y2) in tqdm(enumerate(train_loader)):
                            x, y, y2 = x.to(device), y.to(device), y2.to(device)
                            with torch.no_grad():  # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
                                image_embedding = net.image_encoder(x)
                            pred = self.medsam_inference(net, image_embedding)
                            loss = nn.MSELoss()(pred, y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(net.state_dict(), model)  # 每喂完一个epoch就直接更新参数
        except Exception as er:
            print('run 函数error',er)
            print('报错行数：',er.__traceback__.tb_lineno)
        self._signal.emit('微调结束，已完成所有epoch') # 进度条完成的标志
        print('线程结束！')


class client1(QWidget): # 继承QWidget/QMainWindow这个类别

    def __init__(self):
        super().__init__() # 继承类
        self.initUI()

    def initUI(self):
        splash = QSplashScreen(QPixmap('pix/cover.png')) # 快闪封面
        splash.show()
        time.sleep(2.5)
        mini_font = QFont()
        mini_font.setFamily('微软雅黑')
        mini_font.setPointSize(8)
        self.mini_font = mini_font

        label_font = QFont()
        label_font.setFamily('微软雅黑')
        label_font.setBold(True)
        label_font.setPointSize(18)
        # label_font.setWeight(50)
        self.label_font_Chinese = label_font

        label_font = QFont()
        label_font.setFamily("Roman times")
        label_font.setBold(True)
        label_font.setPointSize(20)
        # label_font.setWeight(50)
        self.label_font_English = label_font

        placeholder_font = QFont()
        placeholder_font.setFamily('微软雅黑')
        placeholder_font.setPointSize(10)
        placeholder_font.setBold(True)
        self.placeholder_font = placeholder_font

        # QToolTip.setFont(QFont("Roman times", 15))
        QToolTip.setFont(placeholder_font)
        # self.setToolTip('<b>奇怪的吴小志同学</b>')
        # 窗口参数的设定
        palette1 = QPalette()
        palette1.setColor(self.backgroundRole(), QColor(255,255,255))  # 背景颜色
        self.setPalette(palette1)
        self.setAutoFillBackground(True)

        self.setFixedSize(1500, 800)
        # self.resize(1500, 800)
        self.setWindowOpacity(0.9)  # 窗口设置透明度，1为完全不透明
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.add_picture()
        # self.add_tab_widget()
        self.setWindowTitle('LivSeg MRI Tools')
        self.setWindowIcon(QIcon('pix/logo5.png'))
        self.add_picture()
        self.add_text()
        self.add_combo_box()
        self.add_btn()
        self.add_pbar()
        self.imgIndex = 0
        self.add_lineedit()
        self.savepath = os.getcwd()
        # print('当前工作目录：',self.savepath)

        self.isOpen = False

        # self.show()
    def add_lineedit(self):
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setGeometry(QRect(700, 700, 60, 60))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText(str(self.imgIndex + 1))  # 当前页数的按钮: self.lineEdit
        self.lineEdit.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.lineEdit.setAttribute(Qt.WA_TranslucentBackground)
        self.lineEdit.setStyleSheet("background-color: rgba(255, 255, 255, 0);color:#8a8e8f")
        self.lineEdit.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.lineEdit.show()

    def add_picture(self):
        '''为界面添加图像'''
        palette = QPalette()
        self.pic = QPixmap("pix/33.jpg")
        # self.pic = self.pic.scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding) # 缩放至正好大小
        palette.setBrush(QPalette.Background, QBrush(self.pic))
        self.setPalette(palette)
        # label = QLabel(self)
        # label.setGeometry(QRect(1100, 500, 500, 500))
        # label.setText('')
        # label.setPixmap(QPixmap('pix/girl.jpeg'))
        # label.setScaledContents(True) # 开启自适应大小
        #
        # label = QLabel(self)
        # label.setGeometry(QRect(50, 0, 300, 800))
        # label.setText('')
        # label.setPixmap(QPixmap('pix/flower.jpeg'))
        # label.setScaledContents(True)  # 开启自适应大小

    def add_text(self):
        label = QLabel(self)
        # 设置标签的左边距，上边距，宽，高
        label.setGeometry(QRect(100, 129, 250, 45))
        # 设置文本标签的字体和大小，粗细等
        # label.setFont(QFont("Roman times", 20))
        label_font = QFont()
        label_font.setFamily('微软雅黑')
        label_font.setBold(True)
        label_font.setPointSize(18)
        # label_font.setWeight(50)
        label.setFont(self.label_font_Chinese)
        label.setStyleSheet("color: white;")  # 设置文字为红色
        # label.setFont(label_font)
        label.setText("选择模型:")

        label2 = QLabel(self)
        label2.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        label2.setStyleSheet("color:#8a8e8f")
        label2.setObjectName("")
        label2.setText("By Liu, Jim @ ZMIC.Fudan University")
        label2.setGeometry(QRect(100, 55, 400, 45))

        label3 = QLabel(self)
        label3.setFont(QFont("微软雅黑", 10, QFont.Bold))
        label3.setStyleSheet("color:white")
        label3.setText("连通域数量:")
        label3.setGeometry(QRect(100, 190, 150, 30))

        label4 = QLabel(self)
        label4.setFont(QFont("微软雅黑", 10, QFont.Bold))
        label4.setStyleSheet("color:white")
        label4.setText("孔洞填充:")
        label4.setGeometry(QRect(350, 190, 150, 30))

        label5 = QLabel(self)
        label5.setFont(QFont("微软雅黑", 10, QFont.Bold))
        label5.setStyleSheet("color:white")
        label5.setText("标签保存:")
        label5.setGeometry(QRect(570, 190, 150, 30))

        self.label6 = QLabel(self)
        self.label6.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.label6.setStyleSheet("color:yellow")
        self.label6.setText("图像名称")
        self.label6.setGeometry(QRect(0, 650, 1457, 30))
        self.label6.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

        label7 = QLabel(self)
        label7.setFont(QFont("微软雅黑", 10, QFont.Bold))
        label7.setStyleSheet("color:white")
        label7.setText("微调轮数:")
        label7.setGeometry(QRect(1000, 190, 150, 30))


    def add_combo_box(self):
        self.combo_box1 = QComboBox(self)
        self.info = ['Unet','Attenton Unet', 'BayeSeg', 'DeepLab', 'MRI_SAM','ensemble','unet3d']
        for i in self.info:
            self.combo_box1.addItem(i)
        self.combo_box1.setFont(self.label_font_Chinese)
        self.combo_box1.setGeometry(QRect(300, 129, 300, 45))

        self.combo_box2 = QComboBox(self)
        self.info2 = ['1', '2', 'max']
        for i in self.info2:
            self.combo_box2.addItem(i)
        self.combo_box2.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.combo_box2.setGeometry(QRect(230, 190, 80, 30))

        self.combo_box3 = QComboBox(self)
        self.info3 = ['开', '关']
        for i in self.info3:
            self.combo_box3.addItem(i)
        self.combo_box3.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.combo_box3.setGeometry(QRect(450, 190, 80, 30))

        self.combo_box4 = QComboBox(self)
        self.info4 = ['保存', '不保存']
        for i in self.info4:
            self.combo_box4.addItem(i)
        self.combo_box4.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.combo_box4.setGeometry(QRect(670, 190, 100, 30))

        self.combo_box5 = QComboBox(self)
        self.info5 = ['1', '2','4','8','16','32','64','128','256','512','1024']
        for i in self.info5:
            self.combo_box5.addItem(i)
        self.combo_box5.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.combo_box5.setGeometry(QRect(1100, 190, 85, 30))

    def add_pbar(self):
        # 进度条设置
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(1200,135, 200, 30)
        self.pbar.setGeometry(0, 780, 1500, 20)
        self.pbar.setValue(0)

    def add_btn(self):
        self.button1 = QPushButton(qtawesome.icon('fa.hourglass-start', color='yellow'), "选图分割", self)
        self.button1.setCursor(QCursor(Qt.PointingHandCursor))  # 手形按钮点击
        self.button1.setFont(self.label_font_Chinese)
        self.button1.setGeometry(QRect(700, 129, 220, 45))
        self.button1.setStyleSheet("QPushButton{color:white}"
                                   "QPushButton:hover{color:red}"
                                   "QPushButton{background-color:rgb(40,40,255)}"
                                   "QPushButton{border:2px}"
                                   "QPushButton{border-radius:20px}"
                                   "QPushButton{padding:2px 4px}")
        self.button1.clicked.connect(self.img_seg)
        self.button1.show()

        self.button2 = QPushButton(qtawesome.icon('fa.star', color='yellow'), "模型微调", self)
        self.button2.setCursor(QCursor(Qt.PointingHandCursor))  # 手形按钮点击
        self.button2.setFont(self.label_font_Chinese)
        self.button2.setGeometry(QRect(1000, 129, 220, 45))
        self.button2.setStyleSheet("QPushButton{color:white}"
                                   "QPushButton:hover{color:red}"
                                   "QPushButton{background-color:rgb(88,74,231)}"
                                   "QPushButton{border:2px}"
                                   "QPushButton{border-radius:20px}"
                                   "QPushButton{padding:2px 4px}")
        self.button2.clicked.connect(self.tuning)
        self.button2.show()

        self.button3 = QPushButton(qtawesome.icon('fa.download', color='yellow'), "保存路径", self)
        # self.button3 = QPushButton("保存路径", self)
        # self.button3.setIcon(QIcon("pix/saveIcon.jpg"))
        self.button3.setCursor(QCursor(Qt.PointingHandCursor))  # 手形按钮点击
        self.button3.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.button3.setGeometry(QRect(800, 190, 120, 30))
        self.button3.setStyleSheet("QPushButton{color:gray}"
                                   "QPushButton:hover{color:red}"
                                   "QPushButton{background-color:rgb(135,206,250)}"
                                   "QPushButton{border:2px}"
                                   "QPushButton{border-radius:10px}"
                                   "QPushButton{padding:2px 4px}")
        self.button3.clicked.connect(self.set_save_path)
        self.button3.show()


        self.button_up = QPushButton(self)
        self.button_up.setCursor(QCursor(Qt.PointingHandCursor))  # 手形按钮点击
        # self.button_up.setFont(self.label_font_Chinese)
        self.button_up.setGeometry(QRect(800, 700, 60, 60))
        self.button_up.setIcon(QIcon("pix/right.png"))  # 向后翻页的按钮
        self.button_up.setIconSize(QSize(60, 60))
        # self.button_up.setStyleSheet("QPushButton{color:white}"
        #                            "QPushButton:hover{color:red}"
        #                            "QPushButton{background-color:rgb(40,40,255)}"
        #                            "QPushButton{border:2px}"
        #                            "QPushButton{border-radius:10px}"
        #                            "QPushButton{padding:2px 4px}")
        self.button_up.clicked.connect(self.page_up)
        # self.button_up.show()

        self.button_down = QPushButton(self)
        self.button_down.setCursor(QCursor(Qt.PointingHandCursor))  # 手形按钮点击
        # self.button_down.setFont(self.label_font_Chinese)
        self.button_down.setGeometry(QRect(600, 700, 60, 60))
        self.button_down.setIcon(QIcon("pix/left.png"))  # 向后翻页的按钮
        self.button_down.setIconSize(QSize(60, 60))
        # self.button_down.setStyleSheet("QPushButton{color:white}"
        #                            "QPushButton:hover{color:red}"
        #                            "QPushButton{background-color:rgb(40,40,255)}"
        #                            "QPushButton{border:2px}"
        #                            "QPushButton{border-radius:10px}"
        #                            "QPushButton{padding:2px 4px}")
        self.button_down.clicked.connect(self.page_down)
        # self.button_down.show()

    # def set_main(self):
    #     model_name_list = ['Unet','Attenton Unet', 'BayeSeg', 'DeepLab', 'MRI_SAM']
    #     main_model_name = self.combo_box1.currentText()
    #     model_id = self.info.index(main_model_name)
    #     main_model_name2 = model_name_list[model_id] # 找到对应的英文模型名称
    #     model = joblib.load(f'model/{main_model_name2}.pkl') #加载
    #     joblib.dump(model, 'model/model.pkl') # 保存
    #     QMessageBox.about(self,'设置成功',f'{main_model_name}已设置为预测模型！')

    def warning(self):
        QMessageBox.about(self,'温馨提示','抱歉！非试验人员暂不支持此项功能！')
    def img_seg(self): # 选图-分割-预测
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        if dialog.exec():
            self.filenames = dialog.selectedFiles()  # 返回选择文件的名字
            go_on = True
        else:
            QMessageBox.about(self, '读取失败', '请检查是否选择了文件')
            go_on = False # 如果用户没选择任何文件，程序暂停，以保证没有报错
        if go_on == True:
            try:
                model_name_list = ['Unet', 'Attenton Unet', 'BayeSeg', 'DeepLab', 'MRI_SAM','ensemble','unet3d']
                main_model_name = self.combo_box1.currentText()
                topk_info = self.combo_box2.currentText()
                hole_fill_info = self.combo_box3.currentText()
                if_save_info = self.combo_box4.currentText()
                model_id = self.info.index(main_model_name)
                main_model_name2 = model_name_list[model_id]  # 找到对应的英文模型名称
                self.button1.setEnabled(False) # 注意 这个位置让按钮变灰了
                self.thread = Predictle_thread(self.filenames,main_model_name2,topk_info,hole_fill_info,if_save_info,self.savepath) # 这个就是最关键的预测线程
                # 连接信号
                self.thread._signal.connect(self.call_backlog)  # 进程连接回传到GUI的事件
                self.thread._signal2.connect(self.show_img_name)# 进程连接回传到GUI的事件
                # 开始线程
                # self.add_pbar() # 显示进度条
                self.thread.start()
            except Exception as er:
                print(er)
                QMessageBox.about(self, '读取失败', '请检查是否为图像文件、路径中不包含中文！且model文件夹存在模型！')

    def set_save_path(self): # 选图-分割-预测
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory) # 选择文件夹
        if dialog.exec():
            self.filenames = dialog.selectedFiles()  # 返回选择文件的名字
            self.savepath = self.filenames[0]
            QMessageBox.about(self, '路径设置成功', f'预测标签储存路径更改为{self.savepath}')

        else:
            QMessageBox.about(self, '设置失败', '请检查是否选择了文件夹')
            go_on = False # 如果用户没选择任何文件，程序暂停，以保证没有报错

    def call_backlog(self, msg): # 当分割进程结束后负责显示图像
        self.pbar.setValue(int(msg))  # 将线程的参数传入进度条,并且用于控制进程结束时的操作
        try:
            if msg == '100':
                # print('进度条已结束')
                self.img, self.post_mask = self.thread.img,self.thread.post_mask # 把thread里面的图像转移到UI里面
                self.isOpen = True
                self.imgDim = self.img.shape[0]
                if self.imgDim >= 3:
                    self.imgIndex = int(self.imgDim / 2 - 1)  # imgIndex相当于最短边的中点位置（z/2），（x,y,z/2）最能代表这个立体图像
                else:
                    self.imgIndex = int(self.imgDim / 2 - 0.1)

                plt.cla()
                self.fig, self.ax = plt.subplots()
                self.ax.axis('off')
                # print('图像index：',self.imgIndex)
                self.ax.imshow(convert_to_color(self.img[self.imgIndex,:,:]), interpolation='nearest', aspect='auto')
                cavan = FigureCanvas(self.fig)
                tmp = QVBoxLayout()
                tmp.addWidget(cavan)
                # tmp.setContentsMargins(0, 0, 0, 0)

                tmpWidgets = QWidget(self)
                tmpWidgets.setGeometry(250, 210, 450, 450)
                tmpWidgets.setLayout(tmp)
                tmpWidgets.show()

                # plt.cla()
                self.fig2, self.ax2 = plt.subplots()
                self.ax2.axis('off')
                # print('图像index：',self.imgIndex)
                imgtmp = Image.fromarray(convert_to_color(self.img[self.imgIndex,:,:]))
                labeltmp = Image.fromarray(convert_to_red(self.post_mask[self.imgIndex, :, :] * 255))
                self.ax2.imshow(Image.blend(imgtmp, labeltmp, 0.3), interpolation='nearest', aspect='auto')
                cavan2 = FigureCanvas(self.fig2)
                tmp2 = QVBoxLayout()
                tmp2.addWidget(cavan2)
                tmpWidgets2 = QWidget(self)
                tmpWidgets2.setGeometry(750, 210, 450, 450)
                tmpWidgets2.setLayout(tmp2)
                tmpWidgets2.show()
                self.lineEdit.setText(str(self.imgIndex + 1))



                # self.setLayout(tmp)
                # save_arr2nii(self.img, path='tmpimg.nii.gz')
                # save_arr2nii(self.post_mask, path='tmplabel.nii.gz')
                # print(self.img.shape)
                # print(self.post_mask.shape)
                # self.thread.terminate()
                del self.thread # 删除线程
                self.button1.setEnabled(True) # 恢复按钮
                # file_ex = self.filenames[0].split('.')[0]
                if self.combo_box4.currentText() == '保存':
                    QMessageBox.about(self,'预测完成!',f'全部分割预测标签已保存至{self.savepath}中')
                else:
                    QMessageBox.about(self, '预测完成!', f'全部图像已分割完成')
        except Exception as er3:
            print('er3:',er3)
            if self.thread.img == None:
                QMessageBox.about(self, '读取失败!', f'请检查选择的文件全部为图像、路径中不包含中文！且mode文件夹存在模型！')
                del self.thread  # 删除线程
                self.button1.setEnabled(True) # 恢复按钮

    def show_img_name(self, msg):  # 当分割进程结束后负责显示图像
        self.label6.setText(msg)
        if msg == '微调结束，已完成所有epoch':
            del self.thread2  # 删除线程
            self.button2.setEnabled(True)  # 恢复按钮
            QMessageBox.about(self, '微调成功！', f'{self.main_model_name2}模型已更新完毕')
    def tuning(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec():
            self.tuning_dir = dialog.selectedFiles()[0]  # 返回选择目录的名字
        else:
            QMessageBox.about(self, '读取失败', '请检查是否选择了文件')
            return False
        # 接下来要确保，目录下包含img和label，以及对应的文件名相同
        dirlst = os.listdir(self.tuning_dir)
        if 'image' not in dirlst or 'label' not in dirlst:
            QMessageBox.about(self, '读取失败', '您所选择的文件夹下不包含img和label两个文件夹')
            return False
        dir_img = os.path.join(self.tuning_dir,'image')
        dir_label = os.path.join(self.tuning_dir,'label')
        lst_img = sorted(os.listdir(dir_img))
        lst_label = sorted(os.listdir(dir_label))
        if lst_label != lst_img:
            QMessageBox.about(self, '读取失败', 'img和label两个文件夹下文件名不完全一致')
            return False
        try:
            model_name_list = ['Unet', 'Attenton Unet', 'BayeSeg', 'DeepLab', 'MRI_SAM','ensemble','unet3d']
            main_model_name = self.combo_box1.currentText()

            epoch = int(self.combo_box5.currentText())
            model_id = self.info.index(main_model_name)
            self.main_model_name2 = model_name_list[model_id]  # 找到对应的英文模型名称
            if self.main_model_name2 == 'ensemble':
                QMessageBox.about(self, '微调失败', '集成模型不支持微调，请将各个模型分别微调！')
                return False
            self.button2.setEnabled(False)  # 注意 这个位置让按钮变灰了
            self.thread2 = tuning_thread(dir_img,dir_label, self.main_model_name2,epoch)  # 这个就是最关键的预测线程
            # 连接信号
            self.thread2._signal.connect(self.show_img_name)  # 进程连接回传到GUI的事件
            # 开始线程
            self.thread2.start()
        except Exception as er:
            print(er)
            QMessageBox.about(self, '读取失败', '请检查是否为图像文件、路径中不包含中文,以及图像与标签的对应关系')
    def page_up(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex < self.imgDim - 1:
                        self.ax.cla()
                        self.ax2.cla()
                        self.imgIndex += 1
                        self.lineEdit.setText(str(self.imgIndex + 1))
                        self.ax.axis('off')
                        self.ax2.axis('off')
                        self.ax.imshow(convert_to_color(self.img[self.imgIndex,:,:]), interpolation='nearest',
                                       aspect='auto')
                        imgtmp = Image.fromarray(convert_to_color(self.img[self.imgIndex, :, :]))
                        labeltmp = Image.fromarray(convert_to_red(self.post_mask[self.imgIndex, :, :] * 255))
                        self.ax2.imshow(Image.blend(imgtmp, labeltmp, 0.3), interpolation='nearest',
                                       aspect='auto')
                        self.ax.figure.canvas.draw()
                        self.ax2.figure.canvas.draw()
                except Exception as erpup:
                    print(erpup)

        except Exception as err:
            print(err)
    def page_down(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex > 0:
                        self.ax.cla()
                        self.ax2.cla()
                        self.imgIndex -= 1
                        self.lineEdit.setText(str(self.imgIndex + 1))
                        self.ax.axis('off')
                        self.ax2.axis('off')
                        self.ax.imshow(convert_to_color(self.img[self.imgIndex,:,:]), interpolation='nearest',
                                       aspect='auto')
                        imgtmp = Image.fromarray(convert_to_color(self.img[self.imgIndex, :, :]))
                        labeltmp = Image.fromarray(convert_to_red(self.post_mask[self.imgIndex, :, :] * 255))
                        self.ax2.imshow(Image.blend(imgtmp, labeltmp, 0.3), interpolation='nearest',
                                        aspect='auto')
                        self.ax.figure.canvas.draw()
                        self.ax2.figure.canvas.draw()
                except Exception as erpup:
                    print(erpup)
        except Exception as err:
            print(err)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = client1()
    main.show()
    sys.exit(app.exec_())