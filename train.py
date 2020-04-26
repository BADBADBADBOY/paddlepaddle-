from vgg.vgg import *
from resnet.resnet import *
from shufflenet.shufflenet import *
from mobilenet.mobilenet import *

import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
import numpy as np

def optimizer_setting(opti_type,base_lr,decay_step,parameter_list,l2_decay = 5e-4):

    lr = []
    lr = [base_lr * (0.1 ** i) for i in range(len(decay_step) + 1)]
    if(opti_type=='Momentum'):
        optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=fluid.layers.piecewise_decay(
                boundaries=decay_step, values=lr),
                momentum=0.99,
                use_nesterov = True,
                regularization=fluid.regularizer.L2Decay(l2_decay),
                parameter_list=parameter_list)
    elif(opti_type=='SGD'):
        optimizer =fluid.optimizer.SGDOptimizer(
                learning_rate=fluid.layers.piecewise_decay(
                boundaries=decay_step, values=lr),
                parameter_list=parameter_list,
                regularization=fluid.regularizer.L2Decay(l2_decay))
    elif(opti_type=='Adam'):
        optimizer =fluid.optimizer.AdamOptimizer(
                learning_rate=fluid.layers.piecewise_decay(
                boundaries=decay_step, values=lr),
                parameter_list=parameter_list,
                regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer
    
def eval(model, data):
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    for batch_id, data in enumerate(data()):
        dy_x_data = np.array([x[0].reshape(3, 224, 224) for x in data]).astype('float32')
        if len(np.array([x[1] for x in data]).astype('int64')) != batch_size:
            continue
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True

        out = model(img)
        # loss = fluid.layers.cross_entropy(input=out, label=label)
        # avg_loss = fluid.layers.mean(x=loss)

        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

        # dy_out = avg_loss.numpy()

        # total_loss += dy_out
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1

        # print("epoch id: %d, batch step: %d, loss: %f" % (eop, batch_id, dy_out))
        if batch_id % 10 == 0:
            print("test | batch step %d, acc1 %0.3f acc5 %0.3f" % (batch_id, total_acc1 / total_sample, total_acc5 / total_sample))

    print("kpis\ttest_acc1\t%0.3f" % (total_acc1 / total_sample))
    print("kpis\ttest_acc5\t%0.3f" % (total_acc5 / total_sample))
    print("final eval acc1 %0.3f acc5 %0.3f" % (total_acc1 / total_sample, total_acc5 / total_sample))

epochs = 100
batch_size = 32
img_num = 6080
seed = 8888
base_lr = 0.1
opti_type = 'SGD'
decay_step = [30,50,80]

place = fluid.CUDAPlace(0)
with fluid.dygraph.guard(place):
    
    np.random.seed(seed)
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed

    model = MobileNetV3_Large(num_classes = 102)

    train_reader = paddle.batch(paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size)

    test_reader = paddle.batch(paddle.dataset.flowers.test(use_xmap=False), batch_size=batch_size)
    
    
    decay_step = [(img_num//batch_size) * step for step in decay_step]
    
    optimizer = optimizer_setting(opti_type,base_lr,decay_step,model.parameters(),l2_decay = 5e-4)
    
    for eop in range(epochs):

            model.train()
            
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0

            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                if len(np.array([x[1] for x in data]).astype('int64')) != batch_size:
                    continue
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                out = model(img)
                loss = fluid.layers.cross_entropy(input=out, label=label)
                avg_loss = fluid.layers.mean(x=loss)

                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

                dy_out = avg_loss.numpy()

                avg_loss.backward()

                optimizer.minimize(avg_loss)
                model.clear_gradients()

                total_loss += dy_out
                total_acc1 += acc_top1.numpy()
                total_acc5 += acc_top5.numpy()
                total_sample += 1

                if batch_id % 10 == 0:
                    print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f" % \
                          (eop, batch_id, total_loss / total_sample, \
                           total_acc1 / total_sample, total_acc5 / total_sample))

            
            print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
            print("kpis\ttrain_acc5\t%0.3f" % (total_acc5 / total_sample))
            print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))
            
            print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f" % \
                  (eop, batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample))

            eval(model, test_reader)

            fluid.save_dygraph(model.state_dict(),'resnet_params')

