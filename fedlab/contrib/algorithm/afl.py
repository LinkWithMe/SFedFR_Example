"""
AFL主动选择策略

"""

import random
import numpy as np
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDSerialClientTrainer
from ...core.standalone import StandalonePipeline
from ...utils import functional as F
class AFL(SyncServerHandler):
    def setup_optim(self, d):
        self.d = d # the number of candidate

    def sample_candidates(self):
        metric = None
        n = 2
        self.alpha1 = 0.75  # 0.75
        self.alpha2 = 0.01  # 0.01
        self.alpha3 = 0.1   # 0.1
        probs = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        num_select = int(self.alpha1 * self.num_clients)
        argsorted_value_list = np.argsort(metric)
        drop_client_idxs = argsorted_value_list[:self.num_clients - num_select]
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        # probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        np.random.seed(0)
        selected = np.random.choice(self.num_clients, num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(self.num_clients)) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')
        return selected_client_idxs.astype(int)

    def sample_clients(self):
        metric = random.sample(range(200),200)
        n = 10
        self.alpha1 = 0.75  # 0.75
        self.alpha2 = 0.01  # 0.01
        self.alpha3 = 0.1   # 0.1
        probs = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        num_select = int(self.alpha1 * self.num_clients)
        argsorted_value_list = np.argsort(metric)
        drop_client_idxs = argsorted_value_list[:self.num_clients - num_select]
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        # probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        np.random.seed(0)
        selected = np.random.choice(self.num_clients, num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(self.num_clients)) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        # print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')
        return selected_client_idxs.tolist()

    def sample_clients_bb(self, candidates, losses,accss):
        """
        BB改，添加了对中间结果的显示
        :param candidates:
        :param losses:
        :param accss:
        :return:
        """


        # print("初始的客户端编号",candidates)
        # print("初始的点火率值",losses)
        sort = np.array(losses).argsort().tolist()
        sort.reverse()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]

        selected_losses = np.array(losses)[sort][0:self.num_clients_per_round]
        selected_accss = np.array(accss)[sort][0:self.num_clients_per_round]

        # 选择前5后5
        # selected_clients_front = np.array(candidates)[sort][0:self.num_clients_per_round-5]
        # selected_clients_last = np.array(candidates)[sort][-5:]
        # selected_clients = np.concatenate((selected_clients_front, selected_clients_last), axis=0)

        # print("选择的客户端编号为",selected_clients)
        return selected_clients.tolist(),selected_losses.tolist(),selected_accss.tolist()


#####################
#                   #
#       Client      #
#                   #
#####################

class AFLClientTrainer(SGDSerialClientTrainer):
    def evaluate(self, id_list, model_parameters):
        self.set_model(model_parameters)
        losses = []
        accss = []
        for id in id_list:
            dataloader = self.dataset.get_dataloader(id,1024)
            loss, acc = F.evaluate(self._model, torch.nn.CrossEntropyLoss(),dataloader)
            losses.append(loss)
            accss.append(acc)
        return losses,accss


    """
    BB改：SNN版本的点火率
    """
    def evaluate_FireRate(self, id_list, model_parameters):
        self.set_model(model_parameters)
        losses = []
        accss = []
        for id in id_list:
            dataloader = self.dataset.get_dataloader(id, 1024)
            loss, acc = F.evaluate_FireRate(self._model, torch.nn.CrossEntropyLoss(),dataloader)
            losses.append(loss)

            # BB添加，返回准确率
            accss.append(acc)

        return losses,accss