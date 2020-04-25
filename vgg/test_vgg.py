"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: test_vgg.py
@time: 2020/4/25 16:39

"""
from vgg_pp import *
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
import numpy as np

def load_premodel_dict(model, premodel_dict):
    model_dict_data = {}

    for name in model.named_sublayers():
        for key in name[1].state_dict().keys():
            new_key = name[0] + '.' + key
            model_dict_data[new_key] = name[1].state_dict()[key]
    new_model_dict = {}
    for key in model_dict_data.keys():
        if (key not in premodel_dict.keys()):
            new_model_dict[key] = model_dict_data[key]
        else:
            new_model_dict[key] = premodel_dict[key]
    return new_model_dict

with fluid.dygraph.guard():
    model = vgg16()
    image = np.ones((1,3,224,224)).astype(np.float32)
    image = to_variable(image)
    out = model(image)
    print(out.shape)
    print(out.numpy()[0][-1])

    ## load premodel
    premodel_dict,_ = fluid.dygraph.load_dygraph('../../vgg16') #这里自动添加后缀
    premodel_dict = load_premodel_dict(model,premodel_dict)
    model.load_dict(premodel_dict)

    out = model(image)
    print(out.shape)
    print(out.numpy()[0][-1])


