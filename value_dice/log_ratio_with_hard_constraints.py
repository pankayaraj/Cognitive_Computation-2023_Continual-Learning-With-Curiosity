from model import Nu_NN
import torch
import numpy as np
from math import exp


class Algo_Param():
    def __init__(self, gamma=0.9):
        self.gamma = gamma


class Log_Ratio_HC():

    def __init__(self, nu_param, algo_param, lamda_lr=0.0003, deterministic_env=True, averege_next_nu = True,
                 discrete_policy=True, save_path = "temp", load_path="temp",
                 continuous_policy_sample_size = 10):

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


    def train_ratio(self, data, target_policy):

            self.debug_V = {"exp":None, "log_exp":None}
            self.update_no += 1

            state = data.state
            action = data.action
            next_state = data.next_state
            initial_state = data.initial_state

            weight = torch.Tensor(self.algo_param.gamma**data.time_step).to(self.nu_param.device)

            #reshaping the weight tensor to facilitate the elmentwise multiplication operation
            no_data = weight.size()[0]
            weight = torch.reshape(weight, [no_data, 1])


            next_action = target_policy.sample(next_state, format="torch")
            initial_action = target_policy.sample(initial_state, format="torch")

            nu, next_nu, initial_nu = self.compute(state, action, next_state, next_action, initial_state, initial_action, target_policy)


            delt_nu = nu - self.algo_param.gamma*next_nu

            unweighted_nu_loss_1 = self.f(delt_nu)
            unweighted_nu_loss_2 = initial_nu


            loss_1 = torch.log(torch.sum(weight*unweighted_nu_loss_1)/torch.sum(weight))
            loss_2 = torch.sum(weight*unweighted_nu_loss_2)/torch.sum(weight)



            self.debug_V["exp"] = torch.sum(weight*unweighted_nu_loss_1)/torch.sum(weight)
            self.debug_V["log_exp"] = loss_1
            self.debug_V["linear"] = loss_2

            #loss_1 = torch.log(torch.sum(unweighted_nu_loss_1))
            #loss_2 = torch.sum(unweighted_nu_loss_2)

            neg_KL = loss_1 - (1-self.algo_param.gamma)*loss_2
            #neg_KL_unweigheted = torch.log(torch.sum(unweighted_nu_loss_1)) - (1 - self.algo_param.gamma) * torch.sum(unweighted_nu_loss_2)



            #main function
            loss = neg_KL

            self.nu_optimizer.zero_grad()
            loss.backward()
            self.nu_optimizer.step()



            if self.update_no%self.target_hard_update_interval == 0:
                self.hard_update(source=self.nu_network, target=self.nu_network_target)

            self.current_KL = (neg_KL.item())

    def debug(self):
        return self.debug_V["exp"], self.debug_V["log_exp"], self.debug_V["linear"]

    def compute(self, state, action, next_state, next_action, initial_state, initial_action, target_policy, policy=None):

        nu = self.nu_network(state, action)
        initial_nu = self.nu_network(initial_state, initial_action)

        #in case of the deterministic env then we can take the averge over all actions n it would suffice. Even in the absese
        #if averge netx nu flag is set we average over all the actions to reduce any bias

        if self.average_next_nu and self.discrete_poliy:
            # This is only for discrete policy. Since the hot vector dim corresponds to the no of actions in this case
            batch_size = torch.Tensor(state).size()[0]

            one_hot_next_action = [torch.zeros([batch_size, self.nu_param.action_dim])
                                   for _ in range(self.nu_param.action_dim)]
            for action_i in range(self.nu_param.action_dim):
                one_hot_next_action[action_i][:, action_i] = 1

            next_target_probabilities = target_policy.get_probabilities(next_state)
            all_next_nu = [self.nu_network(next_state, one_hot_next_action[action_i])
                           for action_i in range(self.nu_param.action_dim)]

            next_nu = torch.zeros([batch_size, 1]).to(self.nu_param.device)
            for action_i in range(self.nu_param.action_dim):
                next_nu += torch.reshape(next_target_probabilities[:, action_i], shape=(batch_size, 1)) * all_next_nu[
                    action_i]

        elif self.average_next_nu and self.discrete_poliy == False:

            batch_size = torch.Tensor(state).size()[0]
            """
            """
            #here we sample the actor many times to average
            all_next_action = [policy.sample(next_state) for i in range(self.continuous_policy_sample_size)]
            all_next_nu = [self.nu_network(next_state, all_next_action[action_i])
                           for action_i in range(self.continuous_policy_sample_size)]
            next_nu = torch.stack(all_next_nu, dim=0).sum(dim=0)/self.continuous_policy_sample_size

        else:
            next_nu = self.nu_network(next_state, next_action)


        return nu, next_nu, initial_nu


    def compute_for_eval(self, state, action, next_state, target_policy, limiter_network=True, policy = None):

        nu_network = self.nu_network_target

        nu = nu_network(state, action)

        # temporary fix for None in next state
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_state)), device=self.nu_param.device, dtype=torch.bool)

        non_final_next_states = torch.Tensor([s for s in next_state if s is not None]).to(self.nu_param.device)
        next_action = target_policy.sample(non_final_next_states)

        #in case of the deterministic env then we can take the averge over all actions n it would suffice. Even in the absese
        #if averge netx nu flag is set we average over all the actions to reduce any bias

        if self.average_next_nu and self.discrete_poliy:
            # This is only for discrete policy. Since the hot vector dim corresponds to the no of actions in this case
            batch_size_n = non_final_next_states.size()[0]

            one_hot_next_action = [torch.zeros([batch_size_n, self.nu_param.action_dim]).to(self.nu_param.device)
                                   for _ in range(self.nu_param.action_dim)]
            for action_i in range(self.nu_param.action_dim):
                one_hot_next_action[action_i][:, action_i] = 1

            next_target_probabilities = target_policy.get_probabilities(non_final_next_states)
            all_next_nu = [nu_network(non_final_next_states, one_hot_next_action[action_i])
                           for action_i in range(self.nu_param.action_dim)]

            next_nu = torch.zeros([batch_size_n, 1]).to(self.nu_param.device)
            for action_i in range(self.nu_param.action_dim):
                next_nu += torch.reshape(next_target_probabilities[:, action_i], shape=(batch_size_n, 1)) * all_next_nu[
                    action_i]


        elif self.average_next_nu and self.discrete_poliy == False:

            batch_size = torch.Tensor(state).size()[0]
            """
            """
            #here we sample the actor many times to average
            all_next_action = [policy.sample(next_state) for i in range(self.continuous_policy_sample_size)]
            all_next_nu = [self.nu_network(next_state, all_next_action[action_i])
                           for action_i in range(self.continuous_policy_sample_size)]
            next_nu = torch.stack(all_next_nu, dim=0).sum(dim=0)/self.continuous_policy_sample_size

        else:
            next_nu = nu_network(non_final_next_states, next_action)

        #at final state next state is None. So the corresponding nu value is left as zero
        batch_size = np.shape(state)[0]
        next_nu_temp = torch.zeros(batch_size, device=self.nu_param.device).unsqueeze(1)
        next_nu_temp[non_final_mask] = next_nu
        next_nu = next_nu_temp


        return nu, next_nu



    def get_log_state_action_density_ratio(self, data, target_policy, limiter_network=True):
        #since this is just the evaluation and don't need inital state, action we can simply use the q learning memory.

        nu, next_nu= self.compute_for_eval(data.state, data.action, data.next_state, target_policy, limiter_network=limiter_network )
        return nu - self.algo_param.gamma*next_nu

    def hard_update(self, source, target):
        target.load_state_dict(source.state_dict())