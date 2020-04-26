#-*- coding:utf-8 _*-
"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: vgg_pp.py
@time: 2020/4/26 14:41

"""

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D,BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable



class hswish(fluid.dygraph.Layer):
    def forward(self, x):
        out = x * fluid.layers.relu6(x + 3) / 6
        return out

class relu(fluid.dygraph.Layer):
    def forward(self, x):
        out = fluid.layers.relu(x)
        return out

class hsigmoid(fluid.dygraph.Layer):
    def forward(self, x):
        out = fluid.layers.relu6(x + 3) / 6
        return out


class SeModule(fluid.dygraph.Layer):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = fluid.dygraph.Sequential(
            Conv2D(in_size, in_size // reduction, filter_size=1, stride=1, padding=0),
            BatchNorm(in_size // reduction,act='relu'),
            Conv2D(in_size // reduction, in_size, filter_size=1, stride=1, padding=0),
            BatchNorm(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(fluid.dygraph.Layer):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = Conv2D(in_size, expand_size, filter_size=1, stride=1, padding=0, )
        self.bn1 = BatchNorm(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = Conv2D(expand_size, expand_size, filter_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size)
        self.bn2 = BatchNorm(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = Conv2D(expand_size, out_size, filter_size=1, stride=1, padding=0)
        self.bn3 = BatchNorm(out_size)

        self.shortcut = fluid.dygraph.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = fluid.dygraph.Sequential(
                Conv2D(in_size, out_size, filter_size=1, stride=1, padding=0),
                BatchNorm(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(fluid.dygraph.Layer):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = Conv2D(3, 16, filter_size=3, stride=2, padding=1)
        self.bn1 = BatchNorm(16)
        self.hs1 = hswish()

        self.bneck = fluid.dygraph.Sequential(
            Block(3, 16, 16, 16, relu(), None, 1),
            Block(3, 16, 64, 24, relu(), None, 2),
            Block(3, 24, 72, 24, relu(), None, 1),
            Block(5, 24, 72, 40, relu(), SeModule(40), 2),
            Block(5, 40, 120, 40, relu(), SeModule(40), 1),
            Block(5, 40, 120, 40, relu(), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = Conv2D(160, 960, filter_size=1, stride=1, padding=0)
        self.bn2 = BatchNorm(960)
        self.hs2 = hswish()
        self.linear3 = Linear(960, 1280)
        self.bn3 = BatchNorm(1280)
        self.hs3 = hswish()
        self.linear4 = Linear(1280, num_classes,act='softmax')

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = fluid.layers.pool2d(out, 7,'avg')
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(fluid.dygraph.Layer):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = Conv2D(3, 16, filter_size=3, stride=2, padding=1)
        self.bn1 = BatchNorm(16)
        self.hs1 = hswish()

        self.bneck = fluid.dygraph.Sequential(
            Block(3, 16, 16, 16, relu(), SeModule(16), 2),
            Block(3, 16, 72, 24, relu(), None, 2),
            Block(3, 24, 88, 24, relu(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = Conv2D(96, 576, filter_size=1, stride=1, padding=0)
        self.bn2 = BatchNorm(576)
        self.hs2 = hswish()
        self.linear3 = Linear(576, 1280)
        self.bn3 = BatchNorm(1280)
        self.hs3 = hswish()
        self.linear4 = Linear(1280, num_classes,act='softmax')

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = fluid.layers.pool2d(out, 7,'avg')
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



# with fluid.dygraph.guard():
#     net = MobileNetV3_Large()
#     x = np.zeros((1,3,224,224)).astype(np.float32)
#     x = to_variable(x)
#     y = net(x)
#     print(y.shape)

