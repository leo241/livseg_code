import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
from PreProcessing3 import DataProcessor,MyDataset # 图像预处理
from deeplabv3 import DeepLab # 网络结构
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, print_iter=100, first_iter=0,
              loss_func=nn.BCEWithLogitsLoss(), loss_func2=nn.MSELoss(),L2 = 0):
        train_writer = SummaryWriter('logs/train')
        val_writer = SummaryWriter('logs/val')
        net = net.to(device)  # 加入gpu
        # net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=L2)
        i = 0
        stop = False
        best_val_loss = float('inf')
        for epoch in range(EPOCH):
            if stop == True:
                break
            train_loss_lst = [] # 每一轮epoch都记录平均的loss
            val_loss_lst = []
            for step, (x, y, y2) in enumerate(self.train_loader):
                # print(torch.mean(x))
                # print(torch.mean(y))
                # print(y2)
                x, y, y2 = x.to(device), y.to(device), y2.to(device)
                output1 = net(x)
                # print(output.shape) # (batchsize,classnum,l,h)
                # print(y.shape)       # (batchsize,classnum,l,h)
                # print(type(output1),type(y),type(output2),type(y2))
                # print(loss_func(output1, y))
                # print(loss_func2(output2,y2))
                output1 = output1.to(torch.float)
                y = y.to(torch.float)
                # output2 = output2.to(torch.float)
                y2 = y2.to(torch.float)
                loss1 = loss_func(output1, y)
                loss = loss1
                # loss = loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                train_loss_lst.append(float(loss1))
                if i % print_iter == 0:
                    print(f'\n epoch:{epoch + 1}\niteration: {i + first_iter}')
                    print('\n tmp train loss:', float(loss))
            print(f'第{epoch+1}轮结束，训练损失平均为{np.mean(train_loss_lst)}')
            train_writer.add_scalar('loss', np.mean(train_loss_lst), epoch+1) # 记录这一轮的平均train loss
            for data in self.valid_loader:
                x_valid, y_valid, slice_valid = data
                x_valid, y_valid, slice_valid = x_valid.to(device), y_valid.to(device), slice_valid.to(device)
                output1 = net(x_valid)
                valid_loss = loss_func(output1, y_valid)
                val_loss_lst.append(float(valid_loss))
            val_loss_tmp = np.mean(val_loss_lst)
            print(f'验证损失平均为{val_loss_tmp}')
            val_writer.add_scalar('loss', val_loss_tmp, epoch+1)
            torch.save(net.state_dict(), f'final.pth')
            if val_loss_tmp < best_val_loss:
                torch.save(net.state_dict(), f'best.pth')
                best_val_loss = val_loss_tmp

if __name__ == '__main__':
    '''设置部分超参数'''
    batch_size = 2
    L2 = 1e-4 # 正则化参数
    depth = 4 # Unet网络的深度
    EPOCH = 200
    net = DeepLab(num_classes=2,pretrained=False)
    # net.load_state_dict(torch.load(r'D:\study\pga\newtry0427\code practice\实验记录\deeplab_DWI_1e-4.pth')) # 预训练载入
    trains, vals, tests = single_domain_dataset('lianying') # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = cross_modality_dataset()
    # trains, vals, tests = domain_generalization_dataset(domain1='feilipu',domain2 = 'lianying', ratio=[1, 1])  # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = trains[0:1], vals[:1], tests[:1] # 小样本测试
    dp = DataProcessor(PIXEL = 256)
    train_list, valid_list, test_list = dp.get_data(trains,vals,tests)  # 获取训练集，验证集，测试集上的数据（暂时以列表的形式）
    # def check(id): # 这个可以放在论文里做可视化
    #     img, label = train_list[id][0], train_list[id][1]
    #     plt.imshow(Image.blend(img, label, 0.5)x)
    #     plt.show()
    #     plt.imshow(img)
    #     plt.show()
    #     print(train_list[id])
    # check(132)
    print(len(train_list), len(valid_list), len(test_list))


    train_data = MyDataset(train_list)
    valid_data = MyDataset(valid_list)  # 从image2tentor
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)


    unet_processor = nn_processor(train_loader, valid_loader)
    unet_processor.train(net, EPOCH=EPOCH, lr=0.001,print_iter=1000,L2=L2)