# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .basic_dataset import FedDataset, Subset
from ...utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner
from ...utils.functional import partition_report


class PartitionedMNIST(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        partition (str, optional): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root,
                 path,
                 num_clients,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 major_classes_num=3,
                 seed=None,
                 transform=None,
                 unblance=False,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform
        self.unblance = unblance

        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            major_classes_num=major_classes_num,
                            unblance=unblance,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   unblance=False,
                   major_classes_num=3,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        trainset = torchvision.datasets.MNIST(root=self.root,
                                                train=True,
                                                download=download)

        """
        BB添加：非平衡数据集构造
        """
        if unblance:
            # 提取特征和标签
            X, y = trainset.data, trainset.targets

            # 选择少数和多数类别
            minority_classes = [0, 1, 2, 3, 4]
            majority_classes = [5, 6, 7, 8, 9]

            # 找出少数类别和多数类别的索引
            minority_indices = np.where(np.isin(y, minority_classes))[0]
            majority_indices = np.where(np.isin(y, majority_classes))[0]

            # 确定多数类别样本数量
            num_minority_samples = len(minority_indices)
            num_majority_samples_needed = int(num_minority_samples * (1 / 3))  # 3:1的比例

            # 随机选择多数类别的样本，并复制多次，直到达到所需的不平衡比例
            selected_majority_indices = np.random.choice(majority_indices, size=num_majority_samples_needed, replace=True)
            balanced_indices = np.concatenate((minority_indices, selected_majority_indices))

            # 根据索引重新组织数据集
            X_balanced = X[balanced_indices]
            y_balanced = y[balanced_indices]

            # 输出不平衡数据集的样本数量
            print("Balanced dataset size:", len(X_balanced))

            # （如果需要）将重新组织的数据集保存为MNIST数据集的样式
            trainset.data = trainset.data[balanced_indices]
            trainset.targets = trainset.targets[balanced_indices]





        partitioner = MNISTPartitioner(trainset.targets,
                                        self.num_clients,
                                        partition=partition,
                                        dir_alpha=dir_alpha,
                                        verbose=verbose,
                                        major_classes_num=major_classes_num,
                                        seed=seed)

        # partition
        subsets = {
            cid: Subset(trainset,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

        """
               BB添加：分区后生成报告和图片
               """
        print("你已经执行到了分区步骤！")
        # csv_file = "../partition-reports//noniid-labeliid/10个客户端base.csv"
        csv_file = "../partition-reports//Shards/MNIST_Shards3_200客户端.csv"
        partition_report(trainset.targets, partitioner.client_dict,
                         class_num=10,
                         verbose=False, file=csv_file)
        noniid_labeldir_part_df = pd.read_csv(csv_file, header=1)
        noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
        col_names = [f"class{i}" for i in range(10)]
        for col in col_names:
            noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(
                int)

        # select first 10 clients for bar plot
        noniid_labeldir_part_df[col_names].plot.barh(stacked=True)
        # plt.tight_layout()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('sample num')
        plt.show()
        # plt.savefig("../partition-reports//noniid-labeliid/10个客户端base.png")
        plt.savefig("../partition-reports//Shards/MNIST_Shards3_200客户端.png")

    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
