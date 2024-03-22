from json import load
import os
import argparse
import random
from copy import deepcopy
from munch import Munch
import sys
sys.path.append("../")
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.contrib.algorithm import SyncServerHandler, SGDSerialClientTrainer
from fedlab.models.cnnsnn import MNISTNet

from fedlab.utils.functional import evaluate, setup_seed

args = Munch()
args.total_client = 100
args.com_round = 100
args.sample_ratio = 1
args.batch_size = 128
args.epochs = 1
args.lr = 0.005

args.preprocess = True
args.seed = 0
args.alpha = 0.3
args.d= 10

setup_seed(args.seed)
test_data = torchvision.datasets.MNIST(root="../datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=1024)

model = MNISTNet()

handler = SyncServerHandler(model=model,
                            global_round=args.com_round,
                            sample_ratio=args.sample_ratio)
trainer = SGDSerialClientTrainer(model, args.total_client, cuda=True)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# mnist = PathologicalMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist", num_clients=args.total_client, shards=200)
mnist = PartitionedMNIST(root='../datasets/mnist/',
                         path="../datasets/mnist/iid-demo/",
                         num_clients=args.total_client,
                         partition="iid",
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
handler.num_clients = trainer.num_clients
while handler.if_stop is False:
    if round%150==0:
        args.lr=args.lr*0.5

    sampled_clients = handler.sample_clients()
    broadcast = handler.downlink_package

    # client side
    trainer.local_process_BB(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)

    accuracy.append(acc)
    losslist.append(loss)
    if(round %10 ==0):
        torch.save(model, './SNN/model.pth')
        torch.save(model.state_dict(), './SNN/model_params.pth')
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}".format(round, acc, max(accuracy)))
    #     if acc>=0.97:
    #         break
    round += 1
end_time = time.time()
print((end_time - begin_time) / 60.0)
torch.save(accuracy, "./exp_logs/{}, accuracy_{}_B{}_S{}_R{}_Seed{}_T{}.pkl".format(args.alg, "mnist", args.batch_size,
                                                                                    args.sample_ratio, args.com_round,
                                                                                    args.seed,
                                                                                    time.strftime("%Y-%m-%d-%H:%M:%S")))