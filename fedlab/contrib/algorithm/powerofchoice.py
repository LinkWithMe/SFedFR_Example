import csv
import random
import numpy as np
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDSerialClientTrainer
from ...core.standalone import StandalonePipeline
from ...utils import functional as F


#####################
#                   #
#      Pipeline     #
#                   #
#####################

class PowerofchoicePipeline(StandalonePipeline):
    def main(self):
        while self.handler.if_stop is False:
            candidates = self.handler.sample_candidates()
            losses = self.trainer.evaluate(candidates,
                                           self.handler.model_parameters)

            # server side
            sampled_clients = self.handler.sample_clients(candidates, losses)
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)


#####################
#                   #
#       Server      #
#                   #
#####################


class Powerofchoice(SyncServerHandler):
    def setup_optim(self, d):
        self.d = d # the number of candidate

    def sample_candidates(self):
        selection = random.sample(range(self.num_clients), self.d)
        selection = sorted(selection)
        #print("候选名单为", selection)
        return selection

    def sample_clients(self, candidates, losses):
        # print("初始的客户端编号",candidates)
        # print("初始的点火率值",losses)
        sort = np.array(losses).argsort().tolist()
        sort.reverse()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]

        # 选择前5后5
        # selected_clients_front = np.array(candidates)[sort][0:self.num_clients_per_round-5]
        # selected_clients_last = np.array(candidates)[sort][-5:]
        # selected_clients = np.concatenate((selected_clients_front, selected_clients_last), axis=0)

        # print("选择的客户端编号为",selected_clients)
        return selected_clients.tolist()

    def sample_clients_bb(self, candidates, losses,accss):
        """
        BB改，添加了对中间结果的显示
        :param candidates:
        :param losses:
        :param accss:
        :return: 选择客户端，选择客户端loss，选择客户端准确率
        """


        # print("初始的客户端编号",candidates)
        # print("初始的点火率值",losses)
        sort = np.array(losses).argsort().tolist()
        sort.reverse()

        # 正常选择
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        selected_losses = np.array(losses)[sort][0:self.num_clients_per_round]
        selected_accss = np.array(accss)[sort][0:self.num_clients_per_round]

        # 选择第2个
        # selected_clients = np.array(candidates)[sort][1:self.num_clients_per_round]
        # selected_losses = np.array(losses)[sort][1:self.num_clients_per_round]
        # selected_accss = np.array(accss)[sort][1:self.num_clients_per_round]

        # 选择第三个
        # selected_clients = np.array(candidates)[sort][2:self.num_clients_per_round]
        # selected_losses = np.array(losses)[sort][2:self.num_clients_per_round]
        # selected_accss = np.array(accss)[sort][2:self.num_clients_per_round]


        # 选择前2后3
        # selected_clients_front = np.array(candidates)[sort][0:self.num_clients_per_round-5]
        # selected_clients_last = np.array(candidates)[sort][-4:-7]
        # selected_clients = np.concatenate((selected_clients, selected_clients_last), axis=0)

        # print("选择的客户端编号为",selected_clients)
        return selected_clients.tolist(),selected_losses.tolist(),selected_accss.tolist()

    def sample_clients_SelectUpdate(self, candidates, firerateBefore, firerateAfter):
        """
        BB改，
        :param candidates:
        :param losses:
        :param accss:
        :return: 选择客户端，选择客户端loss，选择客户端准确率
        """
        firerateDiff = [a - b for a, b in zip(firerateAfter, firerateBefore)]
        firerateDiffAbs = [abs(x) for x in firerateDiff]
        sort = np.array(firerateDiffAbs).argsort().tolist()
        sort.reverse()

        # 正常选择
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        selectedDiff = np.array(firerateDiff)[sort][0:self.num_clients_per_round]

        return selected_clients.tolist(), selectedDiff.tolist()

    def sample_clients_SelectUpdate_noabs(self, candidates, firerateBefore, firerateAfter):
        """
        BB改，
        :param candidates:
        :param losses:
        :param accss:
        :return: 选择客户端，选择客户端loss，选择客户端准确率
        """
        firerateDiff = [a - b for a, b in zip(firerateAfter, firerateBefore)]
        sort = np.array(firerateDiff).argsort().tolist()
        sort.reverse()

        # 正常选择
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        selectedDiff = np.array(firerateDiff)[sort][0:self.num_clients_per_round]

        return selected_clients.tolist(), selectedDiff.tolist()



#####################
#                   #
#       Client      #
#                   #
#####################

class PowerofchoiceSerialClientTrainer(SGDSerialClientTrainer):

    def csv_reader(self, filename):
        amount_list = []

        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                amount = int(row['Amount'])
                amount_list.append(amount)

        # print(amount_list)
        return amount_list
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


