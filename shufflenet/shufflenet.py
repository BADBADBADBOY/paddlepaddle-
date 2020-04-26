"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: shufflenet.py
@time: 2020/4/25 20:06

"""
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import paddle
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = fluid.layers.reshape(x,[batchsize, groups,channels_per_group, height, width])

    x =  fluid.layers.transpose(x,[0,2,1,3,4])
    # flatten
    x = fluid.layers.reshape(x,[batchsize, -1, height, width])

    return x


class InvertedResidual(fluid.dygraph.Layer):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = fluid.dygraph.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                BatchNorm(inp),
                Conv2D(inp, branch_features, filter_size=1, stride=1, padding=0),
                BatchNorm(branch_features,act='relu')
            )

        self.branch2 = fluid.dygraph.Sequential(
            Conv2D(inp if (self.stride > 1) else branch_features,
                      branch_features, filter_size=1, stride=1, padding=0),
            BatchNorm(branch_features,act='relu'),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            BatchNorm(branch_features),
            Conv2D(branch_features, branch_features, filter_size=1, stride=1, padding=0),
            BatchNorm(branch_features,act='relu')
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0):
        return Conv2D(i, o, kernel_size, stride, padding, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x_size = x.shape[1]//2
            x1, x2 = x[:,0:x_size],x[:,x_size:]
            out = fluid.layers.concat([x1, self.branch2(x2)], axis=1)
        else:
            out = fluid.layers.concat([self.branch1(x), self.branch2(x)], axis=1)

        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(fluid.dygraph.Layer):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = fluid.dygraph.Sequential(
            Conv2D(input_channels, output_channels, 3, 2, 1),
            BatchNorm(output_channels,act='relu')
        )
        input_channels = output_channels

        self.maxpool = Pool2D(pool_size=3, pool_stride=2, pool_padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, fluid.dygraph.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = fluid.dygraph.Sequential(
            Conv2D(input_channels, output_channels, 1, 1, 0),
            BatchNorm(output_channels,act='relu')
        )
        import math
        stdv = 1.0 / math.sqrt(512 * 1.0)

        import math
        stdv = 1.0 / math.sqrt(output_channels * 1.0)
        self.fc =  Linear(output_channels,
                   num_classes,
                   act='softmax',
                   param_attr=fluid.param_attr.ParamAttr(
                   initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = fluid.layers.reduce_mean(x,[2,3])  # globalpool
        x = self.fc(x)
        return x


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)

# import numpy as np
# from paddle.fluid.dygraph import to_variable
# with fluid.dygraph.guard():
#     model = shufflenet_v2_x0_5()
#     image = np.ones((1,3,224,224)).astype(np.float32)
#     image = to_variable(image)
#     out = model(image)
#     print(out.shape)
#     print(out.numpy()[0][-1])