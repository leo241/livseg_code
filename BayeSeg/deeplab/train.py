import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset_divide2 import single_domain_dataset,domain_generalization_dataset # 数据集划分
from PreProcessing2 import DataProcessor,MyDataset # 图像预处理
from BayesSeg2 import BayeSeg,BayeSeg_Criterion # 网络结构
import argparse
from args2 import add_management_args, add_experiment_args, add_bayes_args
# from postprocessing import top_k_connected_postprocessing # mask后处理
# from evaluate_metrics import dice_score,RAVD,assd # 评价指标
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, net,criterion, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, print_iter=100, first_iter=0,
              loss_func=nn.BCEWithLogitsLoss(), loss_func2=nn.MSELoss(),L2 = 0):
        train_writer = SummaryWriter('logs/train')
        val_writer = SummaryWriter('logs/val')
        # print(1)
        net = net.to(device)  # 加入gpu
        # net.train() # 这个我先去掉了
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=L2)
        # print(2)
        i = 0
        stop = False
        best_val_loss = float('inf')
        # print(3)
        for epoch in range(EPOCH):
            if stop == True:
                break
            train_loss_lst = [] # 每一轮epoch都记录平均的loss
            val_loss_lst = []
            # print('epoch right')
            for step, (x, y, y2) in tqdm(enumerate(self.train_loader)):
                # print('step done')
                # print(torch.mean(x))
                # print(torch.mean(y))
                # print(y2)
                x, y, y2 = x.to(device), y.to(device), y2.to(device)
                output1 = net(x)

                # output1 = output1.to(torch.float)
                # y = y.to(torch.float)

                losses, loss_dict = criterion(output1, y)
                loss = losses
                # loss = loss1
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                i += 1
                train_loss_lst.append(float(loss))
                if i % print_iter == 0:
                    print(f'\n epoch:{epoch + 1}\niteration: {i + first_iter}')
                    print('\n tmp train loss:', float(loss))
            print(f'第{epoch+1}轮结束，训练损失平均为{np.mean(train_loss_lst)}')
            train_writer.add_scalar('loss', np.mean(train_loss_lst), epoch+1) # 记录这一轮的平均train loss
            for data in self.valid_loader:
                x_valid, y_valid, slice_valid = data
                x_valid, y_valid, slice_valid = x_valid.to(device), y_valid.to(device), slice_valid.to(device)
                output1 = net(x_valid)
                valid_loss, loss_dict = criterion(output1, y_valid)
                # valid_loss = loss_func(output1, y_valid)
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
    batch_size = 1
    L2 = 1e-5 # 正则化参数 1e-4
    lr = 1e-4 # 3e-4 1e-3
    depth = 4 # Unet网络的深度
    EPOCH = 50
    parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()
    args.bayes_loss_coef = 0 # 更改贝叶斯损失的权重
    net = BayeSeg(args)
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    criterion = BayeSeg_Criterion(args)
    net.load_state_dict(torch.load(r'D:\study\pga\newtry0427\code practice\实验记录\BayeSeg_deeplab_T1_1e-4.pth')) # 预训练载入
    # trains, vals, tests = single_domain_dataset('DWI') # 选择单域或者域泛化做数据集划分
    trains, vals, tests = domain_generalization_dataset(domain1='feilipu',domain2 = 'lianying', ratio=[1, 9])  # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = trains[0:1], trains[:1], trains[:1] # 小样本测试
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
    unet_processor.train(net,criterion, EPOCH=EPOCH, lr=lr,print_iter=1000,L2=L2)