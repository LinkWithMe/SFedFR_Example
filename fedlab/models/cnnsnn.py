"""
BB写
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly import visualizing
from spikingjelly.clock_driven.encoding import PoissonEncoder
class SNNCNN_CIFAR10_BB(nn.Module):
    def __init__(self,T,th):
        super(SNNCNN_CIFAR10_BB, self).__init__()
        self.T = T
        self.th = th
        self.conv1 = layer.Conv2d(3, 6, kernel_size=5, padding=1)
        self.bn1 = layer.BatchNorm2d(6)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(kernel_size=2)


        self.conv2 = layer.Conv2d(6, 16, kernel_size=5, padding=1)
        self.bn2 = layer.BatchNorm2d(16)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool2 = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()
        self.fc1 = layer.Linear(16 * 5 * 5, 120)
        self.sn3 = neuron.IFNode(v_threshold=1.)

        self.fc2 = layer.Linear(120, 100)
        self.sn4 = neuron.IFNode(v_threshold=1.)

        self.fc3 = layer.Linear(100,84)
        self.sn5 = neuron.IFNode(v_threshold=1.)

        self.fc4 = layer.Linear(84, 50)
        self.sn6 = neuron.IFNode(v_threshold=1.)

        self.fc5 = layer.Linear(50, 10)
        self.sn7 = neuron.IFNode(v_threshold=1.)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')





    # 初始化权重
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
        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool2(x)


        x= self.flat(x)
        x= self.fc1(x)
        x= self.sn3(x)
        x= self.fc2(x)
        x= self.sn4(x)
        x= self.fc3(x)
        x= self.sn5(x)
        x= self.fc4(x)
        x= self.sn6(x)
        x= self.fc5(x)
        x= self.sn7(x)

        x=x.mean(0)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet_CIFAR10_SNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10_SNN, self).__init__()
        self.T = 12
        self.features = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2),
            layer.Conv2d(64, 192, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2),
            layer.Conv2d(192, 384, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(384, 256, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2),
        )
        self.fl = layer.Flatten()
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            layer.Linear(256 * 2 * 2, 4096),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            #nn.Dropout(),
            layer.Linear(4096, 4096),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(4096, num_classes),
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
        # print("输入的格式为：",x.shape)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.features(x_seq)
        x_seq = self.fl(x_seq)
        # x_seq = x_seq.view(12165120, 256 * 2 * 2)
        x_seq = self.classifier(x_seq)
        fr = x_seq.mean(0)
        return fr



class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.T = 12
        # Convolutional layers
        self.conv1 = layer.Conv2d(1, 32, 3, 1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.max1 = layer.MaxPool2d(2, 2)
        self.conv2= layer.Conv2d(32, 64, 3, 1)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.max2 = layer.MaxPool2d(2, 2)  # 7 * 7

        self.fl = layer.Flatten()

        # Fully connected layers
        self.fc1 = layer.Linear(1600, 128)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(128, 10)  # 10 classes for MNIST
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.ATan())
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
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(self.conv1(x))

        x = self.max1(x)

        x = self.sn2(self.conv2(x))

        x = self.max2(x)

        x = self.fl(x)  # Flatten the tensor

        x = self.sn3(self.fc1(x))


        x = self.sn4(self.fc2(x))
        # print("最后一层的脉冲为:",x)

        return x.mean(0)




class CSNN_FMNIST(nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
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
        # print("输入的格式为：",x.shape)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        # print("运行到这里！")
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]






class SNNCNN_FMNIST_BB(nn.Module):
    def __init__(self,T,th,only_digits=False):
        super(SNNCNN_FMNIST_BB, self).__init__()
        self.T = T
        self.th = th
        self.conv1 = layer.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)


        self.conv2 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(64)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(kernel_size=2)
        self.dropout1 = layer.Dropout(0.25)

        self.flat = layer.Flatten()

        # self.fc1 = layer.Linear(9216, 128)
        self.fc1 = layer.Linear(12544, 128)
        self.sn3 = neuron.IFNode(v_threshold=1.)
        self.dropout2 = layer.Dropout(0.5)

        self.fc2 = layer.Linear(128,10 if only_digits else 62)
        self.sn4 = neuron.IFNode(v_threshold=1.)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')
    # 初始化权重
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
        # print("初始格式为：",x.shape)
        x = self.conv1(x)
        #　print("卷积1之后：", x.shape)
        x =self.bn1(x)
        # print("归一化1之后：", x.shape)
        x=self.sn1(x)
        # print("脉冲1之后：", x.shape)
        x=self.conv2(x)
        # print("卷积2之后：", x.shape)
        x=self.bn2(x)
        # print("归一化2之后：", x.shape)
        x=self.sn2(x)
        # print("脉冲2之后：", x.shape)
        x = self.pool1(x)
        # print("最大池化之后：", x.shape)
        x = self.dropout1(x)
        # print("随机失活之后：", x.shape)
        x= self.flat(x)
        # print("flat之后：", x.shape)
        x= self.fc1(x)
        # print("全连接1之后：", x.shape)
        x= self.sn3(x)
        # print("脉冲3之后：", x.shape)
        x= self.dropout2(x)
        x= self.fc2(x)
        x= self.sn4(x)
        x=x.mean(0)
        # print("最后的结果格式为",x.shape)
        # print("-----------")
        return x



class SNNCNN_CIFAR10_BBFedCor(nn.Module):
    def __init__(self,T,tau,v_threshold):
        super(SNNCNN_CIFAR10_BBFedCor, self).__init__()
        self.T = T
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset=0.0
        self.th=1.0
        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(kernel_size=2)


        self.conv2 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(64)
        self.sn2 = neuron.IFNode(v_threshold=self.th)


        self.conv3 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(64)
        self.sn3 = neuron.IFNode(v_threshold=self.th)

        self.flat = layer.Flatten()

        # self.fc1 = layer.Linear(9216, 128)
        self.fc1 = layer.Linear(16384, 64)
        self.sn4 = neuron.IFNode(v_threshold=self.th)

        self.fc2 = layer.Linear(64,10 )
        self.sn5 = neuron.IFNode(v_threshold=self.th)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')
    # 初始化权重
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
        # print("初始格式为：",x.shape)
        x = self.conv1(x)
        x =self.bn1(x)
        x=self.sn1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.sn2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.sn3(x)
        x= self.flat(x)
        # print("flat之后：", x.shape)
        x= self.fc1(x)
        # print("全连接1之后：", x.shape)
        x= self.sn4(x)
        # print("脉冲3之后：", x.shape)
        x= self.fc2(x)
        x= self.sn5(x)
        x=x.mean(0)
        return x

class Self_CIFAR10_CifarNet(nn.Module):
    """
    BB改写：
    源文件：Cifar_Net.py  CifarNet类
    """
    def __init__(self, tau, T, v_threshold, v_reset=0.0, dropout_rate=0.7):
        super(Self_CIFAR10_CifarNet, self).__init__()
        self.tau =tau
        self.T=T
        self.v_threshold=v_threshold
        self.v_reset =  v_reset
        self.th=1.0

        self.encoder = PoissonEncoder()

        self.conv1 = layer.Conv2d(3, 64, kernel_size=3, padding=2, stride=4, bias=False)
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.pool1 = layer.MaxPool2d(3,2)


        self.conv2 = layer.Conv2d(64, 192,kernel_size=5, padding=2, bias=False)
        self.bn2 = layer.BatchNorm2d(192)
        self.sn2 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.pool2 = layer.MaxPool2d(3,2)

        self.conv3 = layer.Conv2d(192, 256,kernel_size=3, padding=1, bias=False)
        self.bn3 = layer.BatchNorm2d(256)
        self.sn3 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.pool3 = layer.MaxPool2d(3, 2)

        self.flat = layer.Flatten()
        self.dropout1=layer.Dropout(dropout_rate)

        # self.fc1 = layer.Linear(9216, 2048)
        self.fc1 = layer.Linear(256, 2048,bias=False)
        self.sn4 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.dropout2 = layer.Dropout(dropout_rate)

        self.fc2 = layer.Linear(2048,256,bias=False)
        self.sn5 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)

        self.fc3 = layer.Linear(256, 10,bias=False)
        self.sn6 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)

        # 初始化权重
        self._initialize_weights()


    # 初始化权重
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
        #for t in range(self.T):
        x = self.encoder(x).float()



        x = self.conv1(x)
        x =self.bn1(x)
        x=self.sn1(x)
        # x=self.pool1(x)



        x=self.conv2(x)
        x=self.bn2(x)
        x=self.sn2(x)
        x = self.pool2(x)


        x=self.conv3(x)
        x=self.bn3(x)
        x=self.sn3(x)
        x = self.pool3(x)

        x= self.flat(x)
        x= self.dropout1(x)

        x= self.fc1(x)
        x= self.sn4(x)
        x=self.dropout2(x)

        x= self.fc2(x)
        x= self.sn5(x)

        x = self.fc3(x)
        x = self.sn6(x)

        x=x.mean(0)
        # print("最后的格式为",x.shape)
        return x
class CifarNet(nn.Module):
    def __init__(self, tau, T, v_threshold, v_reset, dropout_rate=0.7):
        super(CifarNet, self).__init__()
        self.tau =tau
        self.T=T
        self.v_threshold=v_threshold
        self.v_reset = v_reset
        self.encoder = PoissonEncoder()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, padding=2, stride=4, bias=False),
            nn.BatchNorm2d(64),
            # neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(3, 1)   # 16 * 16
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(3, 1)  # 16 * 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(3, 2)  # 8 * 8
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(dropout_rate)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 2048, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 256, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
                    # print("你已经经过参数初始化")
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
                    # print("你已经经过参数初始化")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)
                # print("你已经经过参数初始化")

    def forward(self, x):
        # print(x.shape)
        for t in range(self.T):
            x_0 = self.encoder(x).float()
            x_conv_1 = self.conv1(x_0)
            # print(x_conv_1.shape)
            x_conv_2 = self.conv2(x_conv_1)
            # print(x_conv_2.shape)
            x_conv_3 = self.conv3(x_conv_2)
            # print(x_conv_3.shape)
            x_flatten = self.flatten(x_conv_3)

            x_fc_1 = self.fc1(x_flatten)
            x_fc_2 = self.fc2(x_fc_1)

            if t == 0:
                out_spikes_counter = self.classifier(x_fc_2)
            else:
                out_spikes_counter += self.classifier(x_fc_2)

        return out_spikes_counter / self.T



class SNNCNN_CIFAR10_CifarNet(nn.Module):
    def __init__(self, tau, T, v_threshold, v_reset=0.0, dropout_rate=0.7):
        super(SNNCNN_CIFAR10_CifarNet, self).__init__()
        self.tau =tau
        self.T=T
        self.v_threshold=v_threshold
        self.v_reset =  v_reset
        self.th=1.0

        self.encoder = PoissonEncoder()


        self.conv1 = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=11, padding=2, stride=4, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan())
            #layer.MaxPool2d(3, 2)  # 16 * 16
        )

        self.conv2 = nn.Sequential(
            layer.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            layer.BatchNorm2d(192),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(3, 2)  # 16 * 16
        )

        self.conv3 = nn.Sequential(
            layer.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(3, 2) , # 8 * 8
        )

        self.flatten = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(dropout_rate)
        )

        self.fc1 = nn.Sequential(
            #layer.Linear(9216, 2048, bias=False),
            nn.Linear(256, 2048, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            layer.Linear(2048, 256, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        self.classifier = nn.Sequential(
            layer.Linear(256, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        # 初始化权重
        self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
                    #print("你已经经过参数初始化")
            elif isinstance(m,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
                    #print("你已经经过参数初始化")
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)
                #print("你已经经过参数初始化")


    def forward(self, x):
        #　print("初始数据格式为",x.shape)
        # print("初始数据为", x)
        for t in range(self.T):
            x_0 = self.encoder(x).float()
            #print("编码后的格式为",x.shape)
            x_0=x
            x_conv_1 = self.conv1(x_0)
            #print("卷积1", x.shape)
            x_conv_2 = self.conv2(x_conv_1)
            #print("卷积2", x.shape)
            x_conv_3 = self.conv3(x_conv_2)
            #print("卷积3", x.shape)
            # print("卷积的结果为：", x_conv_3)

            x_flatten = self.flatten(x_conv_3)
            #print("展开后", x.shape)

            x_fc_1 = self.fc1(x_flatten)
            #print("全连接1", x.shape)
            x_fc_2 = self.fc2(x_fc_1)
            #print("全连接2", x.shape)
            # print("最后的结果为：", x_fc_2)

            if t == 0:
                out_spikes_counter = self.classifier(x_fc_2)
            else:
                out_spikes_counter += self.classifier(x_fc_2)
            # print("分类结果为：",out_spikes_counter)

        return out_spikes_counter / self.T
    





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

    # def __init__(self, args, T = 10, spiking_neuron: callable = None, **kwargs):
    def __init__(self,  args, spiking_neuron: callable = None, **kwargs):
        super(CNN_FLANC, self).__init__()

        self.T = args.T
        self.th = 0.5
        basis_fract = 0.125
        net_fract= 0.25
        n_basis = 0.25

        self.head = layer.Conv2d(1, 64, 3, stride=1, padding=1)
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
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        # self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2) # 14

        out_2 = round(128*net_fract)
        self.conv2 = DecomBlock(self.filter_bank_2, out_1, out_2, m2, n2, kernel_size=3, bias=False)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        # self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2) # 7

        # self.classifier = layer.Linear(out_2 * 7 * 7*32, 10)
        self.classifier = layer.Linear(out_2 * 7 * 7, 10)

        self.sn3 = neuron.IFNode(v_threshold=self.th)

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
