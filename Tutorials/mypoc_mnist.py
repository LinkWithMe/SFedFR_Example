

from json import load
import os
import argparse
import random
from copy import deepcopy
from munch import Munch

import sys

sys.path.append("../")
from fedlab.contrib.algorithm.afl import AFL, AFLClientTrainer
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu
import matplotlib.pyplot as plt
from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_MNIST, CNN_CIFAR10,CNN_FEMNIST
from fedlab.contrib.algorithm import ScaffoldServerHandler, ScaffoldSerialClientTrainer, FedProxSerialClientTrainer, \
    FedProxServerHandler
from fedlab.models.cnnsnn import SNNCNN_FMNIST_BB, SNNCNN_CIFAR10_CifarNet, CSNN_FMNIST, CNN_FLANC, MNISTNet
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.dataset.partitioned_fashionmnist_BB import PartitionedFashionMNIST

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.powerofchoice import Powerofchoice, PowerofchoiceSerialClientTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用第二块 GPU
args = Munch()
args.total_client = 200
args.com_round = 300
args.sample_ratio = 0.05
# args.sample_ratio = 1 # 全样本
args.batch_size = 128
args.epochs = 5
args.lr = 0.1

args.preprocess = False
args.seed = int(sys.argv[1])
# args.seed = 1

args.alg = "poc"  # fedavg, fedprox, scaffold, fednova, feddyn,poc
# args.alg = "fedprox"
# optim parameter

args.alpha = 0.5
args.d= 60
args.reg = 3e-4
args.mu = 1

# SNN设置
args.T = 12
args.tau = 2.0
args.v_threshold = 1.0
args.th= 1.0
args.channels = 128

setup_seed(args.seed)
test_data = torchvision.datasets.MNIST(root="../datasets/mnist/",
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=1024)

# model = MLP(784, 10)
# model = SNNCNN_FMNIST_BB(T=args.T,th=args.th)
# model = CNN_FEMNIST()
# model = SNNCNN_CIFAR10_CifarNet(T=args.T,tau=args.tau,v_threshold=args.v_threshold)
model = MNISTNet()
# model = CNN_FLANC(args)

if args.alg == "afl":
    handler = AFL(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    trainer = AFLClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)
if args.alg == "fedprox":
    handler = FedProxServerHandler(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    trainer = FedProxSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)

if args.alg == "scaffold":
    handler = ScaffoldServerHandler(model=model,
                                    global_round=args.com_round,
                                    sample_ratio=args.sample_ratio)
    handler.setup_optim(lr=args.lr)

    trainer = ScaffoldSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

if args.alg == "poc":
    handler = Powerofchoice(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio)
    handler.setup_optim(d=args.d)
    trainer = PowerofchoiceSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# mnist = PathologicalMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist", num_clients=args.total_client, shards=200)
# mnist = PartitionedMNIST(root='../datasets/mnist/',
#                          path="../datasets/mnist/client50_unbalance_alpha0.3/",
#                          # path="../datasets/fmnist/alpha2",
#                          num_clients=args.total_client,
#                          partition="unbalance",
#                          # partition="noniid-labeldir",
#                          dir_alpha=args.alpha,
#                          preprocess=args.preprocess,
#                          transform=transforms.Compose( [transforms.ToPILImage(), transforms.ToTensor()])
#                                 )

mnist = PartitionedMNIST(root='../datasets/mnist/',
                         # path="../datasets/mnist/One-Clinet-Base/",
                         path="../datasets/mnist/client200_Shards3_alpha0.5/",
                         # path="../datasets/mnist/client50_非常不均匀_alpha0.3/",
                         num_clients=args.total_client,
                         partition="noniid-#label", #noniid-labeldir,iid,noniid-#label
                         dir_alpha=args.alpha,
                         major_classes_num = 3,
                         preprocess=args.preprocess,
                         transform=transforms.Compose( [transforms.ToPILImage(), transforms.ToTensor()])
                                )
# mnist.preprocess()
trainer.setup_dataset(mnist)

import time
import os

log_dir = "./exp_logs/"
os.makedirs(log_dir, exist_ok=True)

begin_time = time.time()
round = 1

accuracy = []
losslist = []
client_choicelist= []
sampled_clients_firerate = []
sampled_clients_acc = []

handler.num_clients = trainer.num_clients


# for i in range(1,301):
#     candidates = handler.sample_candidates()
#     print(candidates)

while handler.if_stop is False:


    # if round % 100 == 0:
    #     args.lr = args.lr * 0.5
    #     trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)
    #     # trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    #
    # sampled_clients = handler.sample_clients()
    # broadcast = handler.downlink_package
    #
    # # client side
    # client_choicelist.append(sampled_clients)
    # trainer.local_process(broadcast, sampled_clients)
    # uploads = trainer.uplink_package
    #
    # # server side
    # for pack in uploads:
    #     handler.load(pack)
    #
    # loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)






    # BB添加：设置学习率衰减
    if round % 500 == 0:
        args.lr = args.lr * 0.5
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    # candidates = [i for i in range(200)]
    candidates = handler.sample_candidates()
    # losses,accss = trainer.evaluate(candidates, handler.model_parameters)
    firerate , accss = trainer.evaluate_FireRate(candidates, handler.model_parameters)


    # server side
    # sampled_clients = handler.sample_clients(candidates,  losses)
    sampled_clients,sampled_loss,sampled_acc = handler.sample_clients_bb(candidates, firerate,accss)
    # sampled_clients = candidates # 随机选择
    broadcast = handler.downlink_package


    # client side
    # client_choicelist.append(sampled_clients)
    # sampled_clients_firerate.append(sampled_loss)
    # sampled_clients_acc.append(sampled_acc)


    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)




    accuracy.append(acc)
    losslist.append(loss)
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}, Loss: {:.4f}".format(round, acc, max(accuracy),loss),end=",")
    print("选择的客户端为：",sampled_clients)
    #     if acc>=0.97:
    #         break
    round += 1
    if round == 299:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': model.state_dict(),
            'epoch': round  # 保存当前训练轮数
        }
        torch.save(checkpoint, './model_save/mnist/SNN高电火率_Shards3_localtrain{}_d{}_epoch{}_alpha{}_lr{}_seed{}.pth'
                   .format(args.epochs, args.d, round, args.alpha, args.lr, args.seed))

end_time = time.time()
print((end_time - begin_time) / 60.0)
# torch.save(accuracy, "./SNN_result/fmnist_firerate/fmnist_CSNN_R300_T12_firerate_客户端50_alpha1.pkl".format(args.alg, "mnist", args.batch_size,
#                                                                                     args.sample_ratio, args.com_round,
#                                                                                     args.seed,
#                                                                                     time.strftime("%Y-%m-%d-%H:%M:%S")))

# print("根据点火率选择的结果为：")
# for candidates in client_choicelist:
#     print(candidates)
# with open("./SNN_result/fmnist_firerate/xihuashiyan/非常异构fRand所有层选择的客户端_0.3", "w") as file:
#     for item in client_choicelist:
#         file.write(str(item) + "\n")
# with open("./SNN_result/fmnist_firerate/xihuashiyan/非常异构Rand所有层选择的客户端的点火率_0.3", "w") as file:
#     for item in sampled_clients_firerate:
#         file.write(str(item)  + "\n")
# with open("./SNN_result/fmnist_firerate/xihuashiyan/非常异构Rand所有层选择的客户端的准确率_0.3", "w") as file:
#     for item in sampled_clients_acc:
#         file.write(str(item)  + "\n")



#
# plt.plot(range(1, round), accuracy)
# plt.xlabel('Round')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.savefig("./SNN_result/fmnist_firerate/xihuashiyan/accuracy_fRand所有层_客户端50_alpha0.3非常异构.png")
# plt.show()
# plt.close()
#
#
# plt.plot(range(1, round), losslist)
# plt.xlabel('Round')
# plt.ylabel('loss')
# plt.title('Training loss')
# plt.savefig("./SNN_result/fmnist_firerate/xihuashiyan/loss_Rand所有层_客户端50_alpha0.3非常异构.png")
# plt.show()
# plt.close()
