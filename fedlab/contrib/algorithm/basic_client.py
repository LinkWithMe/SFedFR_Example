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

from copy import deepcopy
import torch
from ...core.client.trainer import ClientTrainer, SerialClientTrainer
from ...utils import Logger, SerializationTool
from spikingjelly.activation_based import functional
import torch.nn as nn

class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """
    def __init__(self,
                 model:torch.nn.Module,
                 cuda:bool=False,
                 device:str=None,
                 logger:Logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr =lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


class SGDSerialClientTrainer(SerialClientTrainer):
    """Deprecated
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    def setup_dataset(self, dataset):
        self.dataset = dataset
    """
    BB改：验证tr师兄想法，计算batch中每个样本的loss，然后生成loss分布，并进行正交
    """
    def setup_optim_BB(self, epochs, batch_size, lr,reg):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr,weight_decay=reg)
        self.criterion = torch.nn.CrossEntropyLoss()
        # BB改
        #　self.criterion = torch.nn.MSELoss()

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        # BB改
        # 　self.criterion = torch.nn.MSELoss()

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)
    def local_process_BB(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train_BB(model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.序列化模型参数
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                # print(loss)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                functional.reset_net(self.model)


        return [self.model_parameters]


    def train_BB(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.序列化模型参数
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        BB改：验证tr师兄想法，计算batch中每个样本的loss，然后生成loss分布，并进行正交
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # print("数据的格式为，",data.shape)

                # 计算单个loss
                losses = []
                output = self.model(data)
                for i in range(data.size(0)):
                    sample_output = output[i].unsqueeze(0)  # 获取单个样本的输出
                    sample_label = target[i].unsqueeze(0)  # 获取单个样本的标签
                    loss = self.criterion(sample_output, sample_label)  # 计算损失
                    losses.append(loss.item())  # 将损失添加到列表中

                # loss = self.criterion(output, target)
                # print(len(losses))
                # print(losses)
                '''
                CVor
                '''
                deps_model = nn.Sequential(
                    nn.Linear(len(losses), 8),
                    nn.Tanh(),
                    nn.Linear(8, 8),
                    nn.Tanh(),
                    nn.Linear(8, len(losses))
                ).cuda()
                losses = torch.tensor(losses).cuda()
                processed_losses = deps_model(losses)

                deps_w = processed_losses.clone()
                deps_w = (deps_w - deps_w.mean()) / (deps_w.std() + 1e-5)
                deps_v = torch.exp(deps_w - deps_w.detach()).mean()
                alpha = - self.compute_correlation_coefficient(deps_w, losses) / deps_w.std()
                CVor = torch.exp(alpha * (torch.exp(deps_v - deps_v.detach()) - torch.exp(deps_w - deps_w.detach())))

                total_loss = torch.dot(CVor, losses)

                self.optimizer.zero_grad()
                total_loss.sum().backward()
                # print(total_loss.sum())
                # for w, w_t in zip(trainable_params(self.model), global_params):
                #     w.grad.data += self.args.mu * (w.data - w_t.data)
                self.optimizer.step()

                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # functional.reset_net(self.model)


        return [self.model_parameters]

    def compute_correlation_coefficient(self, x, y):
        # 计算向量的均值
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # 计算向量与其均值的差
        xm = x - mean_x
        ym = y - mean_y

        # 计算分子，即协方差的和
        numerator = torch.sum(xm * ym)

        # 计算分母，即两个向量的标准差的乘积
        denominator = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))

        # 计算相关系数
        correlation_coefficient = numerator / denominator

        return correlation_coefficient

    def leave_one_out_control_variates(self, losses):
        total_loss = sum(losses)
        loo_losses = [(total_loss - loss) / (len(losses) - 1) for loss in losses]

        # Convert the list of tensor losses to a single tensor
        return torch.stack(loo_losses)
