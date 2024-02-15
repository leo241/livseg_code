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
from PreProcessing import DataProcessor,MyDataset # 图像预处理
# from attention_unet3 import AttentionUNet
from atnunet2 import AttU_Net
from torchvision.models import resnet50
from postprocessing import top_k_connected_postprocessing # mask后处理
from evaluate_metrics import dice_score,RAVD,assd # 评价指标
from tqdm import tqdm

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
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=L2)
        i = 0
        stop = False
        best_val_loss = float('inf')
        for epoch in range(EPOCH):
            if stop == True:
                break
            train_loss_lst = [] # 每一轮epoch都记录平均的loss
            val_loss_lst = []
            for step, (x, y) in tqdm(enumerate(self.train_loader)):
                x, y = x.to(device), y.to(device)
                # x = x.to(torch.float)
                output1= net(x)
                output1 = output1.to(torch.float)
                y = y.to(torch.float)
                loss = loss_func(output1, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                train_loss_lst.append(float(loss))
                if i % print_iter == 0:
                    print(f'\n epoch:{epoch + 1}\niteration: {i + first_iter}')
                    print('\n tmp train loss:', float(loss))
            print(f'第{epoch+1}轮结束，训练损失平均为{np.mean(train_loss_lst)}')
            train_writer.add_scalar('loss', np.mean(train_loss_lst), epoch+1) # 记录这一轮的平均train loss
            for data in self.valid_loader:
                x_valid, y_valid= data
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                # x_valid = x_valid.to(torch.float)
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
            # torch.save(net.state_dict(), f'best.pth')

if __name__ == '__main__':
    '''设置部分超参数'''
    batch_size = 2
    L2 = 1e-4 # 正则化参数
    EPOCH = 200
    # 创建一个ViT图像分割模型实例
    image_size = 256 # 出于内存原因，图像大小要做调整


    net = AttU_Net(9,2)
    # 使用预训练的ResNet-50模型进行初始化
    # resnet = resnet50(pretrained=True)
    # net.vit.load_state_dict(resnet.state_dict(), strict=False)  # 加载ResNet-50的权重到ViT模型
    # 预训练
    net.load_state_dict(torch.load('best.pth'))
    # 将模型设置为训练模式
    # net.train()
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    # trains, vals, tests = single_domain_dataset('T1') # 选择单域或者域泛化做数据集划分
    trains, vals, tests = cross_modality_dataset(ratio=[3,1,1])
    # trains, vals, tests = domain_generalization_dataset(domain1='lianying',domain2 = 'tongyong', ratio=[1, 1])  # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = trains[0:1], vals[:1], tests[:1] # 小样本测试
    dp = DataProcessor(PIXEL = image_size)
    train_list1, valid_list1, test_list1 = dp.get_data(trains,vals,tests)  # 获取训练集，验证集，测试集上的数据（暂时以列表的形式）
    train_list2, valid_list2, test_list2 = dp.get_data2(ratio=[3,1,1])
    train_list = train_list1 + train_list2
    valid_list = valid_list1 + valid_list2
    test_list = test_list1 + test_list2
    # def check(id): # 这个可以放在论文里做可视化
    #     img, label = train_list[id][0], train_list[id][1]
    #     plt.imshow(Image.blend(img, label, 0.5))
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
    unet_processor.train(net, EPOCH=EPOCH, lr=0.001,print_iter=2000,L2=L2)