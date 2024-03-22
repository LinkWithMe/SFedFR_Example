"""
BB写，用于Cifar10的SNN网络
"""
import copy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly import visualizing
from spikingjelly.activation_based.layer import SeqToANNContainer
from spikingjelly.clock_driven.encoding import PoissonEncoder
from copy import deepcopy

class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp).float())
class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, timesteps=12, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        # print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        # print ("***** time step per batchnorm".format(self.batch_num))
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    #     self._initialize_weights()
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv1d, nn.Conv2d)):
    #             torch.nn.init.kaiming_normal_(m.weight.data)
    #             if m.bias is not None:
    #                 torch.nn.init.zeros_(m.bias.data)
    #         elif isinstance(m,nn.Linear):
    #             torch.nn.init.kaiming_normal_(m.weight.data)
    #             if m.bias is not None:
    #                 torch.nn.init.zeros_(m.bias.data)
    #         elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
    #             m.weight.data.fill_(1)
    #             torch.nn.init.zeros_(m.bias.data)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.timesteps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i]+ self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage
class GPTNet3(nn.Module):
    """
    GPT给出的CIFAR10示例
    """
    def __init__(self):
        super(GPTNet3, self).__init__()
        self.T = 12
        self.conv1 = layer.Conv2d(3, 32, 5)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(32, 64, 5)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool2 = layer.MaxPool2d(2, 2)


        self.flat = layer.Flatten()
        self.fc1 = layer.Linear(1600, 128)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(128, 10)
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.ATan())

        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(self.pool(self.conv1(x)))
        x = self.sn2(self.pool2(self.conv2(x)))
        x = self.flat(x)
        x = self.sn3(self.fc1(x))
        x = self.sn4(self.fc2(x))
        return x.mean(0)
class GPTNet2(nn.Module):
    def __init__(self):
        super(GPTNet2, self).__init__()
        self.T = 8
        self.conv1 = layer.Conv2d(3, 32, 3, padding=1)  # 3x3卷积层，输入通道3，输出通道32
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(32, 64, 3, padding=1)  # 3x3卷积层，输入通道32，输出通道64
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.conv3 = layer.Conv2d(64, 128, 3, padding=1)  # 3x3卷积层，输入通道64，输出通道128
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool = layer.MaxPool2d(2, 2)  # 2x2最大池化层
        self.flat = layer.Flatten()
        self.fc1 = layer.Linear(32768, 512)  # 全连接层，输入维度为128*4*4，输出维度为512
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(512, 10)  # 全连接层，输入维度为512，输出维度为10（CIFAR-10有10个类别）
        self.sn5 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self._initialize_weights()
        functional.set_step_mode(self, step_mode='m')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        # print("输入数据的格式为，",x.shape)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        x = self.sn3(self.pool(self.conv3(x)))
        x = self.flat(x)
        x = self.sn4(self.fc1(x))
        x = self.sn5(self.fc2(x))
        return x.mean(0)
class GPTNet(nn.Module):
    """
    GPT给出的CIFAR10示例
    """
    def __init__(self):
        super(GPTNet, self).__init__()
        self.T = 12
        self.conv1 = layer.Conv2d(3, 32, 5)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool = layer.MaxPool2d(2, 2)
        self.flat = layer.Flatten()
        self.fc1 = layer.Linear(6272, 64)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(64, 10)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())

        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(self.pool(torch.relu(self.conv1(x))))
        x = self.flat(x)
        x = self.sn2(self.fc1(x))
        x = self.sn3(self.fc2(x))
        return x.mean(0)
class EnhancedSNNCifar(nn.Module):
    def __init__(self, args):
        super(EnhancedSNNCifar, self).__init__()
        self.T = 12
        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.bn1 = layer.BatchNorm2d(32)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.bn2 = layer.BatchNorm2d(32)
        self.pool1 = layer.MaxPool2d(kernel_size=2)

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)

        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())


        self.bn3 = layer.BatchNorm2d(64)
        self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.bn4 = layer.BatchNorm2d(64)
        self.pool2 = layer.MaxPool2d(kernel_size=2)

        self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.sn5 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.bn5 = layer.BatchNorm2d(128)
        self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.sn6 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.bn6 = layer.BatchNorm2d(128)
        self.pool3 = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()


        self.global_fc1 = layer.Linear(128 * 4 * 4, 128)
        self.global_fc2 = layer.Linear(128, 10)

        # 初始化权重
        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')


    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.bn1(self.sn1(self.conv1(x)))
        x = self.bn2(self.sn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(self.sn3(self.conv3(x)))
        x = self.bn4(self.sn4(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(self.sn5(self.conv5(x)))
        x = self.bn6(self.sn6(self.conv6(x)))
        x = self.pool3(x)

        x = self.flat(x)

        x = self.global_fc1(x)
        x = self.global_fc2(x)

        x = x.mean(0)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)
class CSNN_FMNIST_CIFAR10(nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
            layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(8192, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        # 初始化权重
        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]
class JellyActive(nn.Module):
    def __init__(self, tau, T, v_threshold, v_reset=0.0):
        super(JellyActive, self).__init__()
        self.T = T

        # 输入层
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # 卷积层
        self.conv1 = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 16 * 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 8 * 8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(512 * 8 * 8, 1024, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            layer.Dropout(0.7),
            nn.Linear(1024, 512, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(512, 10, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )

    def forward(self, x):
        x = self.static_conv(x)    # 只进行1次编码
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        conv1_spikes_counter = x1   # 卷积层结果，用于中间层返回信息
        conv2_spikes_counter = x2   # 卷积层结果，用于中间层返回信息
        conv3_spikes_counter = x3   # 卷积层结果，用于中间层返回信息
        conv4_spikes_counter = x4   # 卷积层结果，用于中间层返回信息
        out_spikes_counter = self.fc(x4)

        for t in range(1, self.T):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            conv1_spikes_counter = conv1_spikes_counter + x1
            conv2_spikes_counter = conv2_spikes_counter + x2
            conv3_spikes_counter = conv3_spikes_counter + x3
            conv4_spikes_counter = conv4_spikes_counter + x4
            out_spikes_counter += self.fc(x4)
        return out_spikes_counter /  self.T
class SNN_CIFAR10(nn.Module):
    """
    根据FedLab中CNN_CIFAR10的更爱
    """
    def __init__(self,T):
        super(SNN_CIFAR10, self).__init__()
        self.T = T
        self.th=1.0
        self.conv1 = layer.Conv2d(3, 6, 5)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.pool = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(6, 16, 5)
        self.sn2 = neuron.IFNode(v_threshold=self.th)

        self.flat = layer.Flatten()

        self.fc1 = layer.Linear(1600, 120)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self.fc2 = layer.Linear(120, 84)
        self.sn4 = neuron.IFNode(v_threshold=self.th)
        self.fc3 = layer.Linear(84, 10)
        self.sn5 = neuron.IFNode(v_threshold=self.th)
        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')  # # 设置为多步模式

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.conv1(x)
        x = self.sn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.sn2(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.sn3(x)

        x = self.fc2(x)
        x = self.sn4(x)

        x = self.fc3(x)
        x = self.sn5(x)

        x = x.mean(0)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)
class CifarNet(nn.Module):
    def __init__(self, tau, T, v_threshold, v_reset=0.0, dropout_rate=0.7):
        super(CifarNet, self).__init__()
        self.tau =tau
        self.T=T
        self.v_threshold=v_threshold
        self.v_reset =  v_reset
        self.th=1.0

        self.encoder = PoissonEncoder()

        self.conv1 = layer.Conv2d(3, 64, kernel_size=3, padding=2, stride=4, bias=False)
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(3,1)


        self.conv2 = layer.Conv2d(64, 192,kernel_size=5, padding=2, bias=False)
        self.bn2 = layer.BatchNorm2d(192)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool2 = layer.MaxPool2d(3,1)

        self.conv3 = layer.Conv2d(192, 256,kernel_size=3, padding=1, bias=False)
        # self.conv3 = layer.Conv2d(64, 256,kernel_size=3, padding=1, bias=False)
        self.bn3 = layer.BatchNorm2d(256)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self.pool3 = layer.MaxPool2d(3, 2)

        self.flat = layer.Flatten()
        self.dropout1=layer.Dropout(dropout_rate)

        self.fc1 = layer.Linear(6400, 2048,bias = False)
        # self.fc1 = layer.Linear(192, 2048,bias=False)
        self.sn4 = neuron.IFNode(v_threshold=self.th)
        self.dropout2 = layer.Dropout(dropout_rate)

        self.fc2 = layer.Linear(2048,256,bias=False)
        self.sn5 = neuron.IFNode(v_threshold=self.th)

        self.fc3 = layer.Linear(256, 10,bias=False)
        # self.fc3 = layer.Linear(2048, 10, bias=False)
        self.sn6 = neuron.IFNode(v_threshold=self.th)

        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')  # # 设置为多步模式
    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x=self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        # x = self.pool3(x)

        x = self.flat(x)
        x = self.dropout1(x)

        x = self.fc1(x)
        x = self.sn4(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.sn5(x)

        x = self.fc3(x)
        x = self.sn6(x)

        x = x.mean(0)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)
class SelfCifar10(nn.Module):

    """ Simple network"""

    # def __init__(self,  T = 10, spiking_neuron: callable = None, **kwargs):
    def __init__(self):
        super(SelfCifar10, self).__init__()

        self.T = 16
        self.th = 1.0


        self.head = layer.Conv2d(3, 64, kernel_size=3, padding=2, stride=4, bias=False)

        self.conv1 = layer.Conv2d(64, 192,kernel_size=5, padding=2, bias=False) # 28
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(kernel_size=3, stride=2) # 14

        self.conv2 = layer.Conv2d(192, 256,kernel_size=3, padding=1, bias=False)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool2 = layer.MaxPool2d(kernel_size=3, stride=2)

        self.classifier = layer.Linear(256, 10)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')  # # 设置为多步模式

    def forward(self, x):

        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)


        x = self.head(x)    # (10,32,1,28,28)
        # print('head', x.shape)
        x = self.conv1(x)   # (10,32,64,28,28)
        x = self.sn1(x)     # (10,32,128,28,28)
        x = self.pool1(x)   # (10,32,128,28,28)


        x = self.conv2(x)   # (10,32,128,14,14)
        x = self.sn2(x)     # (10,32,128,14,14)
        x = self.pool2(x)   # (10,32,128,14,14)

        x = torch.flatten(x, 2) # (10,32,128,7,7)

        x = self.classifier(x)  # (10,32,6272)

        x = self.sn3(x)       # (10,32,10)
        x =x.mean(0)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

"""
翔宇模型，用于FMNIST
"""
def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size // 2), groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1,bias=False):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias)

def make_model(args, parent=False):

    return CNN_FLANC(args)
class conv_basis(nn.Module):
    def __init__(self, filter_bank, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = filter_bank
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None
        functional.set_step_mode(self, step_mode='m')
        #print(stride)
    def forward(self, x):
        if self.group == 1:
            conv2d = layer.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                  padding=self.kernel_size//2, step_mode='m')
            conv2d.weight = self.weight
            conv2d.padding = self.kernel_size//2
            conv2d.padding_mode = 'zeros'
            x = conv2d(x)
            # x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        else:
            x_list = []
            for xi in torch.split(x, self.basis_size, dim=2):
                conv2d = layer.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                      padding=self.kernel_size // 2, step_mode='m')
                conv2d.weight = self.weight
                conv2d.padding = self.kernel_size // 2
                conv2d.padding_mode = 'zeros'
                x1 = conv2d(xi)
                # print(1 conv1, x1.shape)
                x_list.append(x1)
            x = torch.cat(x_list, dim=2)
            # print(2 conv1, x.shape)
            # ori
            #print(self.weight.shape)
            # x = torch.cat([F.conv2d(input=xi, weight=self.weight, bias=self.bias, stride=self.stride,
            #                         padding=self.kernel_size//2)
            #                for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s


class DecomBlock(nn.Module):
    def __init__(self, filter_bank, in_channels, out_channels, n_basis, basis_size, kernel_size,
                 stride=1, bias=False, conv=conv3x3):
        super(DecomBlock, self).__init__()
        group = in_channels // basis_size
        modules = [conv_basis(filter_bank,in_channels, basis_size, n_basis, kernel_size, stride, bias)]
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)
        functional.set_step_mode(self, step_mode='m')  # # 设置为多步模式
    def forward(self, x):
        return self.conv(x)


class CNN_FLANC(nn.Module):

    """ Simple network"""
    def __init__(self,  args, spiking_neuron: callable = None, **kwargs):
        super(CNN_FLANC, self).__init__()

        self.T = 4
        self.th = 0.5
        basis_fract = 0.125
        net_fract= 0.25
        n_basis = 0.25

        self.head = layer.Conv2d(3, 64, 3, stride=1, padding=1)
        m1 = round(128*n_basis)
        n1 = round(64*basis_fract)
        self.filter_bank_1 = nn.Parameter(torch.empty(m1, n1, 3, 3))

        m2 = round(128*n_basis)
        n2 = round(128*basis_fract)
        self.filter_bank_2 = nn.Parameter(torch.empty(m2, n2, 3, 3))

        X1 = torch.empty(m1, n1, 3, 3)
        torch.nn.init.orthogonal(X1)
        # torch.nn.init.orthogonal_(X1)  # 解决UserWarning
        self.filter_bank_1.data = copy.deepcopy(X1)
        X2 = torch.empty(m2, n2, 3, 3)
        torch.nn.init.orthogonal(X2)
        # torch.nn.init.orthogonal_(X2)
        self.filter_bank_2.data = copy.deepcopy(X2)

        out_1 = round(128*net_fract)
        self.conv1 = DecomBlock(self.filter_bank_1, 64, out_1, m1, n1, kernel_size=3, bias=False) # 28
        self.sn1 = neuron.LIFNode(v_threshold=self.th)
        # self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2) # 14

        out_2 = round(128*net_fract)
        self.conv2 = DecomBlock(self.filter_bank_2, out_1, out_2, m2, n2, kernel_size=3, bias=False)
        self.sn2 = neuron.LIFNode(v_threshold=self.th)
        # self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2) # 7

        # self.classifier = layer.Linear(out_2 * 7 * 7*32, 10)
        self.classifier = layer.Linear(2048, 10)

        self.sn3 = neuron.LIFNode(v_threshold=self.th)

        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')  # # 设置为多步模式

    def forward(self, x):
        #print(x.shape)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # print('加入时间步:', x.shape)

        x = self.head(x)    # (10,32,1,28,28)
        # print('head', x.shape)
        x = self.conv1(x)   # (10,32,64,28,28)
        x = self.sn1(x)     # (10,32,128,28,28)
        x = self.pool1(x)   # (10,32,128,28,28)
        # print('conv1:', x.shape)

        x = self.conv2(x)   # (10,32,128,14,14)
        x = self.sn2(x)     # (10,32,128,14,14)
        x = self.pool2(x)   # (10,32,128,14,14)
        #print(x.shape)
        # print('conv2:', x.shape)
        # 仿照resnet18，替代x.view
        # if self.avgpool.step_mode == 's':
        #     x = torch.flatten(x, 1)
        # elif self.avgpool.step_mode == 'm':
        x = torch.flatten(x, 2) # (10,32,128,7,7)
        # x = x.view(x.size(1), -1)
        # print('x.view:', x.shape)
        x = self.classifier(x)  # (10,32,6272)
        # print('全连接层:', x.shape)

        x = self.sn3(x)       # (10,32,10)
        # print('网络输出:', x.shape)
        x =x.mean(0)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
        # loss_fun = F.l1_loss()     # SNN L1Loss
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
        # loss_fun = F.mse_loss()  # SNN MSEloss
    else:
        raise NotImplementedError
    return loss_fun

def orth_loss(model, args, para_loss_type='L2'):

    loss_fun = loss_type(para_loss_type)

    loss = 0
    for l_id in range(1,3):
        filter_bank = getattr(model,"filter_bank_"+str(l_id))

        #filter_bank_2 = getattr(block,"filter_bank_2")
        all_bank = filter_bank
        num_all_bank = filter_bank.shape[0]
        B = all_bank.view(num_all_bank, -1)
        D = torch.mm(B,torch.t(B)) # 计算两个矩阵的乘积
        D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).cuda())
        loss = loss + D
    return loss
