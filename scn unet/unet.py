import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)




class UNet(nn.Module):
    def __init__(self
                 ,num_classes # 最终分割有几类像素点
                 ,depth = 4 # unet网络的深度
                 ,size = 512 # 图像大小
                 ,rgb_channel = 1 # 图像rgb通道数
                 ,initial_channel = 8 # 网络最开始的通道数
                 ):
        super(UNet, self).__init__()
        self.depth = depth
        self.c0 = Conv_Block(rgb_channel,initial_channel) # 最开始先把通道数扩大一次
        self.d = DownSample() # 2*2 的池化
        self.c = nn.ModuleList() # channel的变化过程
        for i in range(depth):
            self.c.append(Conv_Block(initial_channel*2**i,initial_channel*2**(i+1)))
        for i in range(depth):
            self.c.append(Conv_Block(initial_channel*2**(depth-i),initial_channel*2**(depth-i-1)))
        self.u = nn.ModuleList() # 向上插值、拼接
        for i in range(depth):
            self.u.append(UpSample(initial_channel*2**(depth-i)))
        self.out = nn.Conv2d(initial_channel,num_classes,1,1,0) # pred mask
        # linear_dim = (size/2**depth) ** 2 * initial_channel * 2 ** depth
        linear_dim = size ** 2 * initial_channel / 2 ** depth
        self.fc1 = nn.Linear(int(linear_dim),int(linear_dim ** 0.5))
        self.fc2 = nn.Linear(int(linear_dim ** 0.5),1)

    def forward(self,x):
        L_list = []
        L = self.c0(x) # 512,512,8
        L_list.append(L)
        for i in range(self.depth):
            L = self.c[i](self.d(L)) # 先做降采样，再改变通道数
            L_list.append(L)
        L_flatten = L.view(L.shape[0], -1)
        tmp = self.fc1(L_flatten)
        id = self.fc2(tmp)
        R = L
        for i in range(self.depth):
            R = self.c[self.depth + i](self.u[i](R,L_list[::-1][i+1]))
        pred = self.out(R)
        return pred,id

if __name__ == '__main__':
    size = 256 # 必须满足Unet的深度小于最大值，即 2 ** depth < size,这里 depth < 8,因为每次降采样池化边长都减小一半
    rgb_channel = 1
    depth = 1
    x=torch.zeros(2,rgb_channel,size,size) # (batch-size, rgb_channel_size,length,height)
    net=UNet(num_classes=2,depth = depth,size = size,rgb_channel = rgb_channel,initial_channel =8) # 做二分类
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    output1,output2 = net(x)
    # print(output1.shape) # (batchsize,class_num,len,height)
    print(output2)
    loss = nn.MSELoss()
    def myloss(x,y):
        return torch.square(x-y)
    print(loss(torch.tensor(-0.04),output2))

