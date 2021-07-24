from model import Nu_NN
import torch
import numpy as np
from math import exp


class Algo_Param():
    def __init__(self, gamma=0.9):
        self.gamma = gamma


class Log_Ratio_from_two_buffer():

    def __init__(self, nu_param, algo_param, lamda_lr=0.0003, deterministic_env=True, averege_next_nu = True,
                 discrete_policy=True, save_path = "temp", load_path="temp",
                 continuous_policy_sample_size = 30):

        self.nu_param = nu_param
        self.algo_param = algo_param

        self.lamda_lr = lamda_lr

        self.target_hard_update_interval = self.algo_param.hard_update_interval
        #log_ratio estimator
        self.nu_network = Nu_NN(nu_param, save_path=save_path, load_path=load_path)

        self.nu_network_target = Nu_NN(nu_param, save_path=save_path, load_path=load_path)

        self.nu_optimizer = torch.optim.Adam(self.nu_network.parameters(), lr=self.nu_param.l_r)

        self.Lamda = torch.zeros(1, requires_grad=True, device=self.nu_param.device)
        self.Lamda_optimizer = torch.optim.Adam([self.Lamda] ,lr=self.lamda_lr)

        self.hard_update(source=self.nu_network, target=self.nu_network_target)


        self.nu_base_lr = self.nu_param.l_r
        self.current_lr = self.nu_base_lr




        self.deterministic_env = deterministic_env
        self.average_next_nu = averege_next_nu
        self.discrete_poliy = discrete_policy

        # exponential fucntion
        self.f = lambda x: torch.exp(x)

        self.current_KL = 0
        self.update_no = 0


        self.continuous_policy_sample_size = continuous_policy_sample_size #this is the number of samples ot take when the policy is continuous


    def train_ratio(self, data1, data2, unweighted=True):

            self.debug_V = {"exp":None, "log_exp":None}
            self.update_no += 1

            state1 = data1.state
            action1 = data1.action
            weight1 = torch.Tensor(self.algo_param.gamma ** data1.time_step).to(self.nu_param.device)

            state2 = data2.state
            action2 = data2.action
            weight2 = torch.Tensor(self.algo_param.gamma ** data2.time_step).to(self.nu_param.device)

            # reshaping the weight tensor to facilitate the elmentwise multiplication operation
            no_data1 = weight1.size()[0]
            no_data2 = weight2.size()[0]
            weight1 = torch.reshape(weight1, [no_data1, 1])
            weight2 = torch.reshape(weight2, [no_data2, 1])

            unweighted_nu_loss_1 = self.f(self.nu_network(state1, action1))
            unweighted_nu_loss_2 = self.nu_network(state2, action2)

            if unweighted:
                loss_1 = torch.log(torch.sum(unweighted_nu_loss_1) / len(data1.state))
                loss_2 = torch.sum(unweighted_nu_loss_2) / len(data2.state)
            else:
                loss_1 = torch.log(torch.sum(weight1 * unweighted_nu_loss_1) / torch.sum(weight1))
                loss_2 = torch.sum(weight2 * unweighted_nu_loss_2) / torch.sum(weight2)

            neg_KL = loss_1 - loss_2

            #main function
            loss = neg_KL

            self.nu_optimizer.zero_grad()
            loss.backward()
            self.nu_optimizer.step()

            if self.update_no%self.target_hard_update_interval == 0:
                self.hard_update(source=self.nu_network, target=self.nu_network_target)

            self.current_KL = (neg_KL.item())
            self.loss1 = loss_1
            self.loss2 = loss_2
            self.u_loss1 = unweighted_nu_loss_1
            self.u_loss2 = unweighted_nu_loss_2
            self.x = self.nu_network(state1, action1)

    def get_KL(self, data1, data2, unweighted=True):
        state1 = data1.state
        action1 = data1.action
        weight1 = torch.Tensor(self.algo_param.gamma ** data1.time_step).to(self.nu_param.device)

        state2 = data2.state
        action2 = data2.action
        weight2 = torch.Tensor(self.algo_param.gamma ** data2.time_step).to(self.nu_param.device)

        # reshaping the weight tensor to facilitate the elmentwise multiplication operation
        no_data1 = weight1.size()[0]
        no_data2 = weight2.size()[0]
        weight1 = torch.reshape(weight1, [no_data1, 1])
        weight2 = torch.reshape(weight2, [no_data2, 1])

        unweighted_nu_loss_1 = self.f(self.nu_network(state1, action1))
        unweighted_nu_loss_2 = self.nu_network(state2, action2)

        if unweighted:
            loss_1 = torch.log(torch.sum(unweighted_nu_loss_1))/len(data1.state)
            loss_2 = torch.sum(unweighted_nu_loss_2)/len(data2.state)
        else:
            loss_1 = torch.log(torch.sum(weight1 * unweighted_nu_loss_1) / torch.sum(weight1))
            loss_2 = torch.sum(weight2 * unweighted_nu_loss_2) / torch.sum(weight2)

        neg_KL = loss_1 - loss_2


        return neg_KL

    def debug(self):
        return self.debug_V["exp"], self.debug_V["log_exp"], self.debug_V["linear"]



    def get_log_state_action_density_ratio(self, data, target_policy, limiter_network=True):
        #since this is just the evaluation and don't need inital state, action we can simply use the q learning memory.

        nu = self.nu_network(data.state, data.action)
        return nu

    def hard_update(self, source, target):
        target.load_state_dict(source.state_dict())
