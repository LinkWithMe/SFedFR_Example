import csv
import random
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from .basic_server import SyncServerHandler
from .basic_client import SGDSerialClientTrainer
from torch.distributions.normal import Normal
from ...core.standalone import StandalonePipeline
from ...utils import functional as F
import GPR
from GPR import Kernel_GPR



#####################
#                   #
#       Server      #
#                   #
#####################


class FedCor(SyncServerHandler):

    def setup_optim(self, d):
        self.d = d # the number of candidate

    def sample_candidates(self):
        selection = random.sample(range(self.num_clients), self.d)
        selection = sorted(selection)
        #print("候选名单为", selection)
        return selection

    def __init__(self, num_users, loss_type='LOO', init_noise=0.01, reusable_history_length=10, gamma=1.0,
                 device=torch.device('cpu')):
        """
        Arguments:
            num_users: Number of users in a Federated Learning setting
            loss_type: {"LOO","MML"}
        """
        super(GPR, self).__init__()
        self.num_users = num_users
        self.loss_type = loss_type
        # sigma_n
        self.noise = torch.tensor(init_noise, device=device)
        self.mu = torch.zeros(num_users, device=device).detach()
        self.discount = torch.ones(num_users, device=device).detach()
        self.data = {}
        self.reusable_history_length = reusable_history_length
        self.gamma = gamma
        self.device = device

    def Covariance(self, ids=None):
        raise NotImplementedError("A GPR class must have a function to calculate covariance matrix")

    def Update_Training_Data(self, client_idxs, loss_changes, epoch):
        """
        The training data should be in the form of : data[epoch] = sample_num x [user_indices, loss_change] (N x 2)
        Thus the data[epoch] is in shape S x N x 2
        """
        data = np.concatenate([np.expand_dims(np.array(client_idxs), 2), np.expand_dims(np.array(loss_changes), 2)], 2)
        self.data[epoch] = torch.tensor(data, device=self.device, dtype=torch.float)
        for e in list(self.data.keys()):
            if e + self.reusable_history_length < epoch:
                self.data.pop(e)  # delete too old data

    def Posteriori(self, data):
        """
        Get the posteriori with the data
        data: given in the form [index,loss]
        return:mu|data,Sigma|data
        """
        data = torch.tensor(data, device=self.device, dtype=torch.float)
        indices = data[:, 0].long()
        values = data[:, 1]

        Cov = self.Covariance()

        Sigma_inv = torch.inverse(Cov[indices, :][:, indices])
        mu = self.mu + ((Cov[:, indices].mm(Sigma_inv)).mm((values - self.mu[indices]).unsqueeze(1))).squeeze()
        Sigma = Cov - (Cov[:, indices].mm(Sigma_inv)).mm(Cov[indices, :])
        return mu.detach(), Sigma.detach()

    def Log_Marginal_Likelihood(self, data):
        """
        MML:
        Calculate the log marginal likelihood of the given data
        data: given in the form S x [index,loss]
        return log(p(loss|mu,sigma,correlation,sigma_n))
        """
        res = 0.0

        for d in data:
            idx = d[:, 0].long()
            val = d[:, 1]
            Sigma = self.Covariance(idx)
            distribution = MultivariateNormal(loc=self.mu[idx], covariance_matrix=Sigma)
            res += distribution.log_prob(val)

        return res

    def Log_LOO_Predictive_Probability(self, data):
        """
        LOO:
        Calculate the Log Leave-One-Out Predictive Probability of the given data
        data: given in the form S x [index,loss]
        return: \sum log(p(y_i|y_{-i},mu,sigma,relation,sigma_n))
        """

        # High efficient algorithm exploiting partitioning
        log_p = 0.0
        for d in data:
            idx = d[:, 0].long()
            val = d[:, 1]
            Sigma_inv = torch.inverse(self.Covariance(idx))
            K_inv_y = (Sigma_inv.mm((val - self.mu[idx]).unsqueeze(1))).squeeze()
            for i in range(len(data)):
                mu = val[i] - K_inv_y[i] / Sigma_inv[i, i]
                sigma = torch.sqrt(1 / Sigma_inv[i, i])
                dist = Normal(loc=mu, scale=sigma)
                log_p += dist.log_prob(val[i])

        return log_p

    def Parameter_Groups(self):
        raise NotImplementedError("A GPR class must have a function to get parameter groups = [Mpar,Spar]")

    def MLE_Mean(self):
        """
        Calculate the weighted mean of historical data
        """
        self.mu = torch.zeros(self.num_users, device=self.device).detach()
        current_epoch = max(self.data.keys())
        cum_gamma = torch.zeros(self.num_users, device=self.device)
        for e in self.data.keys():
            for d in self.data[e]:
                idx = d[:, 0].long()
                val = d[:, 1]
                self.mu[idx] += (self.gamma ** (current_epoch - e)) * val
                cum_gamma[idx] += self.gamma ** (current_epoch - e)

        for g in cum_gamma:
            if g == 0.0:
                g += 1e-6
        self.mu = self.mu / cum_gamma
        return self.mu

    def Train(self, lr=1e-3, llr=1e-3, max_epoches=100, schedule_lr=False, schedule_t=None, schedule_gamma=0.1,
              update_mean=False, verbose=True):
        """
        Train hyperparameters(Covariance,noise) of GPR
        data : In shape as [Group,index,value,noise]
        method : {'MML','LOO','NNP'}
            MML:maximize log marginal likelihood
            LOO:maximize Leave-One-Out cross-validation predictive probability
        """

        matrix_params, sigma_params = self.Parameter_Groups()
        optimizer = torch.optim.Adam([{'params': matrix_params, 'lr': lr},
                                      {'params': sigma_params, 'lr': llr}], lr=lr, weight_decay=0.0)
        if schedule_lr:
            lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer, schedule_t, gamma=schedule_gamma)

        if update_mean:
            self.mu = self.MLE_Mean()
            # print(self.mu)
        current_epoch = max(self.data.keys())
        for epoch in range(max_epoches):
            self.zero_grad()
            loss = 0.0
            for e in self.data.keys():
                if self.loss_type == 'LOO':
                    loss -= self.Log_LOO_Predictive_Probability(self.data[e]) * (self.gamma ** (current_epoch - e))
                elif self.loss_type == 'MML':
                    loss -= self.Log_Marginal_Likelihood(self.data[e]) * (self.gamma ** (current_epoch - e))
                else:
                    raise RuntimeError("Not supported training method!!")
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and verbose:
                print("Train_Epoch:{}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(epoch, torch.mean(
                    torch.diagonal(self.Covariance())).detach().item(), loss.item()))

            if schedule_lr:
                lr_scd.step()

        return loss.item()

    def Predict_Loss(self, data, priori_idx, posteriori_idx):
        for p in priori_idx:
            if p in posteriori_idx:
                posteriori_idx.remove(p)  # do not predict the measured idx
        mu_p, sigma_p = self.Posteriori(data[priori_idx, :])

        pdist = MultivariateNormal(loc=mu_p[posteriori_idx],
                                   covariance_matrix=sigma_p[posteriori_idx, :][:, posteriori_idx])
        predict_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx, 1], device=self.device, dtype=torch.float))
        predict_loss = predict_loss.detach().item()
        return predict_loss, mu_p, sigma_p

    def sample_clients(self, number=10, epsilon=0.0, weights=None, Dynamic=False, Dynamic_TH=0.0):
        """
        Select the clients which may lead to the maximal loss decrease
        Sequentially select the client and update the postieriori
        """

        def max_loss_decrease_client(client_group, Sigma, weights=None):
            Sigma_valid = Sigma[:, client_group][client_group, :]
            Diag_valid = self.discount[client_group] / torch.sqrt(torch.diagonal(Sigma_valid))  # alpha_k/sigma_k

            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid, dim=0) * Diag_valid
            else:
                # sum_i Sigma_ik*p_i
                total_loss_decrease = torch.sum(
                    torch.tensor(weights[client_group], device=self.device, dtype=torch.float).reshape(
                        [len(client_group), 1]) * Sigma_valid, dim=0) * Diag_valid
            mld, idx = torch.max(total_loss_decrease, 0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma - Sigma[:, selected_idx:selected_idx + 1].mm(Sigma[selected_idx:selected_idx + 1, :]) / (
            Sigma[selected_idx, selected_idx])

            return selected_idx, p_Sigma, mld.item()

        prob = np.random.rand(1)[0]
        if prob < epsilon:
            # use epsilon-greedy
            return None
        else:
            Sigma = self.Covariance()
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for i in range(number):
                idx, Sigma, total_loss_decrease = max_loss_decrease_client(remain_clients, Sigma, weights)
                if Dynamic and -total_loss_decrease < Dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)

            return selected_clients





#####################
#                   #
#       Client      #
#                   #
#####################

class FedCorSerialClientTrainer(SGDSerialClientTrainer):

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
