import sys

from munch import Munch



sys.path.append("../")
from fedlab.models.cnnsnn import AlexNet_CIFAR10_SNN
import torch
from torch import nn
from fedlab.models.cnn import *
from fedlab.contrib.algorithm.afl import AFL, AFLClientTrainer
from fedlab.contrib.algorithm import FedProxServerHandler, FedProxSerialClientTrainer
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from fedlab.models.spiking_resnet_flanc import SpikingResNet, spiking_ResNet18_FLANC

import matplotlib.pyplot as plt
from fedlab.models.SNNCifar10 import *
from spikingjelly.activation_based import layer, neuron, functional, surrogate
from fedlab.contrib.dataset.partitioned_cifar10_BB import PartitionedCIFAR10
import os
from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.powerofchoice import Powerofchoice, PowerofchoiceSerialClientTrainer

args = Munch()
args.total_client = 200
args.com_round = 300
args.sample_ratio = 0.05
args.batch_size = 128
args.epochs = 10
args.lr = 0.05
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第二块 GPU
args.preprocess = False
args.seed = int(sys.argv[1])
args.alg = "poc"
args.alpha = 0.5
args.d= 10

setup_seed(args.seed)
test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
download=True,
                                       # transform=transforms.ToTensor(),
                                       transform = transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
                                       ])

)

test_loader = DataLoader(test_data, batch_size=1024)
model = AlexNet_CIFAR10_SNN()
# model = SNNCNN_CIFAR10_BBFedCor(T=args.T,tau=args.tau,v_threshold=args.v_threshold)
# model = CNN_CIFAR10_BBFedCor()
# model = SNNCNN_CIFAR10_CifarNet(T=args.T,tau=args.tau,v_threshold=args.v_threshold)
# model = CifarNet(T=args.T,tau=args.tau,v_threshold=args.v_threshold,v_reset=args.v_reset)
# model = SelfCifar10()
# model = CNN_FLANC(args)
# model = spiking_resnet18()
# model = SNN_CIFAR10(T = args.T)
# model = JellyActive(T=args.T,tau=args.tau,v_threshold=args.v_threshold,v_reset=args.v_reset)
# model = CSNN_FMNIST_CIFAR10(T=args.T, channels=args.channels)
# model = EnhancedSNNCifar(args)
# model = GPTNet()
# model =  spiking_ResNet18_FLANC(args, pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

if args.alg == "fedprox":
    handler = FedProxServerHandler(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    trainer = FedProxSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)

if args.alg == "afl":
    handler = AFL(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    trainer = AFLClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)



if args.alg == "poc":
    handler = Powerofchoice(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio)
    handler.setup_optim(d=args.d)
    trainer = PowerofchoiceSerialClientTrainer(model, args.total_client, cuda=True)
    #trainer.setup_optim_BB(args.epochs, args.batch_size, args.lr,args.reg)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# mnist = PathologicalMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist", num_clients=args.total_client, shards=200)
cifar = PartitionedCIFAR10(root='../datasets/cifar10/',
                           path="../datasets/cifar10/client200_Shards3/",
                         num_clients=args.total_client,
                         # partition="unbalance",
                         partition="noniid-#label",
                         dir_alpha=args.alpha,
                         preprocess=args.preprocess,
                         # transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
                                                       ])
                           )
# mnist.preprocess()
trainer.setup_dataset(cifar)

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
while handler.if_stop is False:


    # if round % 100 == 0:
    #     args.lr = args.lr * 0.5
    #     trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)
    #     # trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    #
    # sampled_clients = handler.sample_clients()
    # client_choicelist.append(sampled_clients)
    # broadcast = handler.downlink_package
    #
    # # client side
    # trainer.local_process(broadcast, sampled_clients)
    # uploads = trainer.uplink_package
    #
    # # server side
    # for pack in uploads:
    #     handler.load(pack)
    #
    # loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)





    if round % 300 == 0:
        args.lr = args.lr * 0.5
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)


    candidates = handler.sample_candidates()
    # losses,accss = trainer.evaluate(candidates, handler.model_parameters)
    # firerate, accss = trainer.evaluate_FireRate(candidates, handler.model_parameters)

    # server side
    # sampled_clients = handler.sample_clients(candidates, losses)
    # sampled_clients, sampled_loss, sampled_acc = handler.sample_clients_bb(candidates, firerate, accss)
    sampled_clients = candidates
    broadcast = handler.downlink_package

    # client side
    # client_choicelist.append(sampled_clients)
    # sampled_clients_firerate.append(sampled_loss)
    # sampled_clients_acc.append(sampled_acc)


    # trainer.local_process(broadcast, sampled_clients)
    trainer.local_process(broadcast, candidates)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
    # loss,acc = trainer.evaluate(candidates, handler.model_parameters)


    accuracy.append(acc)
    losslist.append(loss)
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}, Loss: {:.4f}".format(round, acc, max(accuracy),loss),end=",")
    print("最后选择的客户端为：",sampled_clients)
    #     if acc>=0.97:
    #         break
    round += 1

    if  round % 100 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': model.state_dict(),
            'epoch': round  # 保存当前训练轮数
        }
        torch.save(checkpoint, './model_save/cifar10/SNN随机_Shards_localtrain{}_d{}_epoch{}_alpha{}_lr{}_seed{}_client{}.pth'
                   .format(args.epochs,args.d,round,args.alpha,args.lr,args.seed,args.total_client))
end_time = time.time()
print((end_time - begin_time) / 60.0)

#
# with open("./SNN_result/cifar10_firerate/firerate2选择的客户端_0.3_random1", "w") as file:
#     for item in client_choicelist:
#         file.write(str(item) + "\n")
# with open("./SNN_result/cifar10_firerate/firerate2选择的客户端的点火率_0.3_random1", "w") as file:
#     for item in sampled_clients_firerate:
#         file.write(str(item)  + "\n")
# with open("./SNN_result/cifar10_firerate/firerate2选择的客户端的准确率_0.3_random1", "w") as file:
#     for item in sampled_clients_acc:
#         file.write(str(item)  + "\n")
#
#
# plt.plot(range(1, round), accuracy)
# plt.xlabel('Round')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.savefig("./SNN_result/cifar10_client50_alpha0.1/accary_firerate2_alpha0.3_random1.png")
# plt.show()
# plt.close()
#
# plt.plot(range(1, round), losslist)
# plt.xlabel('Round')
# plt.ylabel('loss')
# plt.title('Training loss')
# plt.savefig("./SNN_result/cifar10_client50_alpha0.1/loss_firerate2_alpha0.3_random1.png")
# plt.show()
# plt.close()