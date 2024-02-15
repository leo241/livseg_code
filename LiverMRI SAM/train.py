import numpy as np
import SimpleITK as sitk
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_divide import single_domain_dataset,domain_generalization_dataset,cross_modality_dataset # 数据集划分
from PreProcessing2 import DataProcessor,MyDataset # 图像预处理
# from unet import UNet # 网络结构
from segment_anything import sam_model_registry
# from postprocessing import top_k_connected_postprocessing # mask后处理
# from evaluate_metrics import dice_score,RAVD,assd # 评价指标

import warnings
warnings.filterwarnings("ignore")  # ignore warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速

class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

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

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, print_iter=100, first_iter=0,
              loss_func=nn.MSELoss(), loss_func2=nn.MSELoss(),L2 = 0):
        train_writer = SummaryWriter('logs/train')
        val_writer = SummaryWriter('logs/val')
        # net = net.to(device)  # 加入gpu
        # net.train()
        optimizer = torch.optim.Adam(net.mask_decoder.parameters(), lr=lr,weight_decay=L2) # 只对decoder部分的参数进行微调
        i = 0
        stop = False
        best_val_loss = float('inf')
        for epoch in range(EPOCH):
            if stop == True:
                break
            train_loss_lst = [] # 每一轮epoch都记录平均的loss
            val_loss_lst = []
            for step, (x, y, y2) in tqdm(enumerate(self.train_loader)):
                x, y, y2 = x.to(device), y.to(device), y2.to(device)
                with torch.no_grad():  # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
                    image_embedding = net.image_encoder(x)
                pred = self.medsam_inference(net, image_embedding)
                # print(pred.shape)
                # print(y.shape)
                loss = loss_func(pred, y)
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
                x_valid, y_valid, slice_valid = data
                x_valid, y_valid, slice_valid = x_valid.to(device), y_valid.to(device), slice_valid.to(device)
                with torch.no_grad():  # 这里在fine-tuning时也要no_grad，因为不能改变encoder的参数
                    image_embedding_v = net.image_encoder(x_valid)
                pred_v = self.medsam_inference(net, image_embedding_v)
                valid_loss = loss_func(pred_v, y_valid)
                # output1, output2 = net(x_valid)
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
    L2 = 1e-4 # 正则化参数
    lr = 3e-5 # 训练SAM的学习率不能太大，否则训不动
    EPOCH = 100
    # 载入预训练的SAM模型
    # MedSAM_CKPT_PATH = r'D:\study\pga\newtry0427\code practice\实验记录\sam_DWI_1e-5.pth'
    MedSAM_CKPT_PATH = r'D:\study\pga\newtry0427\code practice\SAM\medsam_vit_b.pth'
    net = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    net = net.to(device)
    net.train()
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    # net.load_state_dict(torch.load('best.pth')) # 预训练载入
    trains, vals, tests = single_domain_dataset('lianying') # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = cross_modality_dataset()
    # trains, vals, tests = domain_generalization_dataset(domain1='feilipu',domain2 = 'lianying', ratio=[1, 1])  # 选择单域或者域泛化做数据集划分
    # trains, vals, tests = trains[0:1], vals[:1], tests[:1] # 小样本测试
    dp = DataProcessor()
    train_list, valid_list, test_list = dp.get_data(trains,vals,tests)
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
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=True, num_workers=0)


    unet_processor = nn_processor(train_loader, valid_loader)
    unet_processor.train(net, EPOCH=EPOCH, lr=lr,print_iter=1000,L2=L2)