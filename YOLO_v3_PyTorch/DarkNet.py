'''
DarkNet主体有106层，没有全连接层，是“全卷积网络”(Fully Convolutional Network)。
可以通过循环的方式，以很少的代码，批量构建网络。这里全部以“硬编码”形式写出，是为了展示网络结构，用于教学目的。
以下DarkNet定义只需快速浏览即可，不难发现module_list的成员，大多是由卷积层、批归一化、ReLU激活构成的block。
少数特殊层更加值得关注，比如Shortcut, Route, Upsample, Yolo等。
'''

import torch
import torch.nn as nn

import cv2
from collections import OrderedDict

from utils import predict_transform


# 创建“空层”对象，什么都不做，只是为了调整网络结构
class EmptyLayer(nn.Module):
    
    def __init__(self):
        super(EmptyLayer, self).__init__()

# DarkNet网络主体
class DarkNet(nn.Module):
    
    def __init__(self, img_size, num_classes, transform=True):
        super(DarkNet, self).__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        self.module_list = nn.ModuleList()
        self.blocks = []    # 为什么变量名叫blocks，而不是modules?
                            # 因为modules是nn.Module对象已有的属性，避免重名
        
        # 卷积，批归一化，激活：三层构成一个block
        self.module_list.append(
            nn.Sequential(OrderedDict([
                # 卷积层，输入3通道，输出32通道，卷积核3x3，步长1，补一圈0，无偏置
                ("conv_0", nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)),
                # 批归一化，参数与之前的卷积层输出通道数一致
                ("batch_norm_0", nn.BatchNorm2d(32)),
                # LeakyReLU激活函数
                ("leaky_0", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})
        
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_1", nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)),
                ("batch_norm_1", nn.BatchNorm2d(64)),
                ("leaky_1", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})
        
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_2", nn.Conv2d(64, 32, kernel_size=(1,1), stride=(1,1), bias=False)),
                ("batch_norm_2", nn.BatchNorm2d(32)),
                ("leaky_2", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_3", nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_3", nn.BatchNorm2d(64)),
                ("leaky_3", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})
        
        # Shortcut连接，先用空层“占坑”，在forward函数定义中再设置
        self.module_list.append(
            nn.Sequential(OrderedDict([
                    ("shortcut_4", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})
        
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_5", nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1),bias=False)),
                ("batch_norm_5", nn.BatchNorm2d(128)),
                ("leaky_5", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_6", nn.Conv2d(128, 64, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_6", nn.BatchNorm2d(64)),
                ("leaky_6", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_7", nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_7", nn.BatchNorm2d(128)),
                ("leaky_7", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_8", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})
        
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_9", nn.Conv2d(128, 64, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_9", nn.BatchNorm2d(64)),
                ("leaky_9", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_10", nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_10", nn.BatchNorm2d(128)),
                ("leaky_10", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_11", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_12", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1),bias=False)),
                ("batch_norm_12", nn.BatchNorm2d(256)),
                ("leaky_12", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_13", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_13", nn.BatchNorm2d(128)),
                ("leaky_13", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_14", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_14", nn.BatchNorm2d(256)),
                ("leaky_14", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_15", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_16", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_16", nn.BatchNorm2d(128)),
                ("leaky_16", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_17", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_17", nn.BatchNorm2d(256)),
                ("leaky_17", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_18", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_19", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_19", nn.BatchNorm2d(128)),
                ("leaky_19", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_20", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_20", nn.BatchNorm2d(256)),
                ("leaky_20", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_21", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_22", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_22", nn.BatchNorm2d(128)),
                ("leaky_22", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_23", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_23", nn.BatchNorm2d(256)),
                ("leaky_23", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_24", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_25", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_25", nn.BatchNorm2d(128)),
                ("leaky_25", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_26", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_26", nn.BatchNorm2d(256)),
                ("leaky_26", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_27", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_28", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_28", nn.BatchNorm2d(128)),
                ("leaky_28", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_29", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_29", nn.BatchNorm2d(256)),
                ("leaky_29", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_30", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_31", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_31", nn.BatchNorm2d(128)),
                ("leaky_31", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_32", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_32", nn.BatchNorm2d(256)),
                ("leaky_32", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_33", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_34", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_34", nn.BatchNorm2d(128)),
                ("leaky_34", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_35", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_35", nn.BatchNorm2d(256)),
                ("leaky_35", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_36", EmptyLayer()),      
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_37", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,2), padding=(1,1),bias=False)),
                ("batch_norm_37", nn.BatchNorm2d(512)),
                ("leaky_37", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_38", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_38", nn.BatchNorm2d(256)),
                ("leaky_38", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_39", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_39", nn.BatchNorm2d(512)),
                ("leaky_39", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_40", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_41", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_41", nn.BatchNorm2d(256)),
                ("leaky_41", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_42", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_42", nn.BatchNorm2d(512)),
                ("leaky_42", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_43", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_44", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_44", nn.BatchNorm2d(256)),
                ("leaky_44", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_45", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_45", nn.BatchNorm2d(512)),
                ("leaky_45", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_46", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_47", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_47", nn.BatchNorm2d(256)),
                ("leaky_47", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_48", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_48", nn.BatchNorm2d(512)),
                ("leaky_48", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_49", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_50", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_50", nn.BatchNorm2d(256)),
                ("leaky_50", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_51", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_51", nn.BatchNorm2d(512)),
                ("leaky_51", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_52", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_53", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_53", nn.BatchNorm2d(256)),
                ("leaky_53", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_54", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_54", nn.BatchNorm2d(512)),
                ("leaky_54", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_55", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_56", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_56", nn.BatchNorm2d(256)),
                ("leaky_56", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_57", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_57", nn.BatchNorm2d(512)),
                ("leaky_57", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_58", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_59", nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_59", nn.BatchNorm2d(256)),
                ("leaky_59", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_60", nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_60", nn.BatchNorm2d(512)),
                ("leaky_60", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_61", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_62", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(2,2), padding=(1,1),bias=False)),
                ("batch_norm_62", nn.BatchNorm2d(1024)),
                ("leaky_62", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_63", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_63", nn.BatchNorm2d(512)),
                ("leaky_63", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_64", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_64", nn.BatchNorm2d(1024)),
                ("leaky_64", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_65", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_66", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_66", nn.BatchNorm2d(512)),
                ("leaky_66", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_67", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_67", nn.BatchNorm2d(1024)),
                ("leaky_67", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_68", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_69", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_69", nn.BatchNorm2d(512)),
                ("leaky_69", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_70", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_70", nn.BatchNorm2d(1024)),
                ("leaky_70", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_71", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_72", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_72", nn.BatchNorm2d(512)),
                ("leaky_72", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_73", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_73", nn.BatchNorm2d(1024)),
                ("leaky_73", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("shortcut_74", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_75", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_75", nn.BatchNorm2d(512)),
                ("leaky_75", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_76", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_76", nn.BatchNorm2d(1024)),
                ("leaky_76", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_77", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_77", nn.BatchNorm2d(512)),
                ("leaky_77", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_78", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                ("batch_norm_78", nn.BatchNorm2d(1024)),
                ("leaky_78", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_79", nn.Conv2d(1024, 512, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_79", nn.BatchNorm2d(512)),
                ("leaky_79", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_80", nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1, 1),bias=False)),
                ("batch_norm_80", nn.BatchNorm2d(1024)),
                ("leaky_80", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_81", nn.Conv2d(1024, 75, kernel_size=(1,1), stride=(1,1)))
            ])))
        self.blocks.append({'type': 'convolutional'})

        # 检测输出层
        # stride=32, 416x416 -> 13x13
        # 最小的特征映射，检测较大的目标
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("Dectection_82", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'yolo', 'anchors': '116,90,  156,198,  373,326'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("route_83", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'route', 'layers': '-4'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_84", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                ("batch_norm_84", nn.BatchNorm2d(256)),
                ("leaky_84", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        # 上采样层
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("upsample_85", nn.Upsample(scale_factor=2, mode='nearest')),
            ])))
        self.blocks.append({'type': 'upsample'})

        # 路由层，连接上一层与第61层
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("route_86", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'route', 'layers': '-1, 61'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_87", nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                ("batch_norm_87", nn.BatchNorm2d(256)),
                ("leaky_87", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_88", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                ("batch_norm_88", nn.BatchNorm2d(512)),
                ("leaky_88", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_89", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                ("batch_norm_89", nn.BatchNorm2d(256)),
                ("leaky_89", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_90", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                ("batch_norm_90", nn.BatchNorm2d(512)),
                ("leaky_90", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_91", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                ("batch_norm_91", nn.BatchNorm2d(256)),
                ("leaky_91", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_92", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                ("batch_norm_92", nn.BatchNorm2d(512)),
                ("leaky_92", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_93", nn.Conv2d(512, 75, kernel_size=(1, 1), stride=(1, 1)))
            ])))
        self.blocks.append({'type': 'convolutional'})

        # 检测输出层
        # stride=16, 416x416 -> 26x26
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("Dectection_94", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'yolo', 'anchors': '30,61,  62,45,  59,119'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("route_95", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'route', 'layers': '-4'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_96", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                ("batch_norm_96", nn.BatchNorm2d(128)),
                ("leaky_96", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("upsample_97", nn.Upsample(scale_factor=2, mode='nearest')),
            ])))
        self.blocks.append({'type': 'upsample'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("route_98", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'route', 'layers': '-1, 36'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_99", nn.Conv2d(384, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_99", nn.BatchNorm2d(128)),
                ("leaky_99", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_100", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_100", nn.BatchNorm2d(256)),
                ("leaky_100", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_101", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_101", nn.BatchNorm2d(128)),
                ("leaky_101", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_102", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_102", nn.BatchNorm2d(256)),
                ("leaky_102", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_103", nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1),bias=False)),
                ("batch_norm_103", nn.BatchNorm2d(128)),
                ("leaky_103", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_104", nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)),
                ("batch_norm_104", nn.BatchNorm2d(256)),
                ("leaky_104", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("conv_105", nn.Conv2d(256, 75, kernel_size=(1,1), stride=(1,1)))
            ])))
        self.blocks.append({'type': 'convolutional'})

        # 检测输出层
        # stride=8, 416x416 -> 52x52
        # 最大的特征映射，可以检测更小的目标
        self.module_list.append(
            nn.Sequential(OrderedDict([
                ("Dectection_106", EmptyLayer()),
            ])))
        self.blocks.append({'type': 'yolo', 'anchors': '10,13,  16,30,  33,23'})
        
        
    def forward(self, x):
        
        modules = self.blocks
        outputs = {}    # 缓存各层输出结果，shortcut和route层可以使用
        write = 0

        # 遍历module_list的每一个模块
        for i, module in enumerate(modules):
            module_type = module["type"]
            
            # 对于卷积层和上采样层，直接将上一层的输入作为输出
            if module_type == 'convolutional' or module_type == "upsample":
                x = self.module_list[i](x)
                
            elif module_type == "route":
                layers = module["layers"].split(",")    # 切分route层的两个索引值
                layers = [int(a) for a in layers]
                
                # layers只有一个参数，无拼接操作
                # 例如layers=[-4]，则直接取4层前的输出
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                
                # layers有多个参数，有拼接操作
                # layers[0]是相对位置，i+layers[0]得到绝对位置
                # layers[1]是绝对位置，直接作为索引
                else:                        
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[layers[1]]
                    x = torch.cat((map1, map2), 1)
    
            elif module_type == "shortcut":
                from_ = int(module["from"])
                # 将上一层的输出与shortcut目标层的输出直接相加
                x = outputs[i-1] + outputs[i+from_]
                
            elif module_type == "yolo":
                
                if self.transform:
                    anchors = module["anchors"].split(",")
                    anchors = [int(a) for a in anchors]
                    anchors = [(anchors[j], anchors[j+1]) for j in range(0, len(anchors), 2)]
                
                    input_dim = int(self.image_size)
                    num_classes = self.num_classes
                    
                    x = predict_transform(x, anchors, input_dim, num_classes)

                    if not write:
                        prediction = x
                        write = 1
                        
                    else:
                        # 经过predict_transform，各yolo层输出只有第1维（BBox数量）不同，在第1维拼接可以得到全局预测结果
                        prediction = torch.cat((prediction, x), 1)

                else:
                    prediction = None

            # 将输出结果缓存到字典outputs
            outputs[i] = x
        
        # 106, 94, 82都是EmptyLayer，实际的输出来自前一卷积层的特征映射
        return prediction, outputs[105], outputs[93], outputs[81]
