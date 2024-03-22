"""
File: mypoc_cnn_SelectUpdate.py
Author:
Date: 2023/12/21
Description: 在mypoc_cnn的基础上，修改了根据点火率选择客户端的功能
             以前是不进行本地训练直接进行挑选，现在先在本地进行几轮训练之后，再进行上传
"""
import sys
sys.path.append("../")
from munch import Munch
from sklearn.model_selection import train_test_split

from fedlab.contrib.dataset.partitioned_fashionmnist_BB import PartitionedFashionMNIST

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from fedlab.models.cnnsnn import SNNCNN_FMNIST_BB, SNNCNN_CIFAR10_CifarNet, CSNN_FMNIST, CNN_FLANC, MNISTNet, \
    AlexNet_CIFAR10_SNN
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.powerofchoice import Powerofchoice, PowerofchoiceSerialClientTrainer
import time
import os
from fedlab.contrib.dataset.partitioned_cifar10_BB import PartitionedCIFAR10

args = Munch()
args.total_client = 200
args.com_round = 300
args.sample_ratio = 0.05
args.batch_size = 128
args.epochs = 10
args.lr = 0.05
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用第二块 GPU
args.preprocess = False
args.seed = int(sys.argv[1])
args.alg = "poc"
args.alpha = 0.5
args.d= 60

setup_seed(args.seed)
# train_data = torchvision.datasets.MNIST(root="../datasets/mnist/",
#                                        train=True,
#                                        transform=transforms.ToTensor())
#
# # 获取训练集和验证集的索引
# train_indices, val_indices = train_test_split(list(range(len(train_data))), test_size=0.2, random_state=0)
# print("数据的长度为：",len(val_indices))
# val_loader = DataLoader(dataset=train_data, batch_size=len(val_indices),
#                         sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices))
#


test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       # transform=transforms.ToTensor(),
                                       transform = transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
                                       ])

)

test_loader = DataLoader(test_data, batch_size=1024)


model = AlexNet_CIFAR10_SNN()


if args.alg == "poc":
    handler = Powerofchoice(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio)
    handler.setup_optim(d=args.d)
    trainer = PowerofchoiceSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

cifar = PartitionedCIFAR10(root='../datasets/cifar10/',
                           path="../datasets/cifar10/client200_Shards3/",
                         num_clients=args.total_client,
                         # partition="noniid-#label",
                         # partition="noniid-labeldir",
                         partition= "unbalance",
                         dir_alpha=args.alpha,
                         major_classes_num = 3,
                         preprocess=args.preprocess,
                         # transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
                                                       ])
                           )
# mnist.preprocess()
trainer.setup_dataset(cifar)

log_dir = "./exp_logs/"
os.makedirs(log_dir, exist_ok=True)

begin_time = time.time()
round = 1

accuracy = []
losslist = []
client_choicelist= []

allclient = []
allclient_firrate = []
allclient_acc = []

sampled_clients_firerate = []
sampled_clients_acc = []


handler.num_clients = trainer.num_clients
while handler.if_stop is False:

    if round % 300 == 0:
        args.lr = args.lr * 0.5
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    # 选出候选集
    candidates = handler.sample_candidates()
    # 计算点火率
    firerateBefore, accssBefore = trainer.evaluate_FireRate(candidates, handler.model_parameters)
    broadcast = handler.downlink_package




    # 本地训练
    trainer.local_process(broadcast, candidates)

    """
    需要添加的功能：
    原先的语句：trainer.evaluate_FireRate(candidates, handler.model_parameters) 只能使用服务器端模型进行点火率测试
    需要添加：使用客户端本地模型进行点火率测试
    需要解决的问题：客户端模型的存储方式，怎么检索到指定客户端的模型
    """
    # 聚合后结果
    firerateAfter, accssAfter = trainer.evaluate_FireRate(candidates, trainer.model_parameters)
    sampled_clients, firerateDiff = handler.sample_clients_SelectUpdate(candidates, firerateBefore, firerateAfter)



    #  上传模型
    uploads = trainer.uplink_package
    """
    选择部分的客户端进行融合：
    1. 根据sampled_clients和candidates，查找出sampled_clients在candidates中的index
    2. 根据index，在uploads中找出指定客户端的模型，构成uploadsSelect
    3. 上传uploadsSelect
    """
    indices = [candidates.index(value) for value in sampled_clients]
    selected_pack = [uploads[index] for index in indices]

    # server side
    for pack in selected_pack:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)





    accuracy.append(acc)
    losslist.append(loss)
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}, Loss: {:.4f},".format(round, acc, max(accuracy),loss),end = "")
    print("最后选择的客户端为", sampled_clients)



    # print("被选择的客户端数目为：",sampled_clients,end=",")
    # print("初始点火率为：",sampled_firerate,end=",")
    # print("准确率为：", sampled_acc,end=",")
    # print("后被选择的数目为：", sampled_clients2, end=",")
    # print("后准确率:",sampled_acc2,end=",")
    # print("后点火率:",sampled_firerate2)

    if  round % 100 == 0 :
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': model.state_dict(),
            'epoch': round  # 保存当前训练轮数
        }
        torch.save(checkpoint, './model_save/cifar10/snn_SFedFR_Shards3_localtrain{}_d{}_epoch{}_alpha{}_lr{}_seed{}_client{}.pth'
                   .format(args.epochs,args.d,round,args.alpha,args.lr,args.seed,args.total_client))

    round += 1
end_time = time.time()
print((end_time - begin_time) / 60.0)


# with open("./Result1025/mnist/IID选择的客户端", "w") as file:
#     for item in client_choicelist:
#         file.write(str(item) + "\n")
# with open("./Result1025/mnist/IID选择的客户端_firerate", "w") as file:
#     for item in sampled_clients_firerate:
#         file.write(str(item)  + "\n")
# with open("./Result1025/mnist/IID选择的客户端_acc", "w") as file:
#     for item in sampled_clients_acc:
#         file.write(str(item)  + "\n")
# with open("./SNN_result/mnist_firerate/所有选择客户端-0.3", "w") as file:
#     for item in allclient:
#         file.write(str(item)  + "\n")
# with open("./SNN_result/mnist_firerate/所有选择客户端的点火率-0.3", "w") as file:
#     for item in allclient_firrate:
#         file.write(str(item)  + "\n")
#
# with open("./SNN_result/mnist_firerate/所有选择客户端的准确率-0.3", "w") as file:
#     for item in allclient_acc:
#         file.write(str(item)  + "\n")



# plt.plot(range(1, round), accuracy)
# plt.xlabel('Round')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.savefig("./Result1025/mnist/tr.png")
# plt.show()
# plt.close()
#
#
# plt.plot(range(1, round), losslist)
# plt.xlabel('Round')
# plt.ylabel('loss')
# plt.title('Training loss')
# plt.savefig("./Result1025/mnist/tr.png")
# plt.show()
# plt.close()
