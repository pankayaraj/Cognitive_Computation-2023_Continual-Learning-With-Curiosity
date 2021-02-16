import torch
import numpy as np

from q_learning import Q_learning


class Q_learner_Policy():
    def __init__(self, q_function, nn_param):

        self.nn_params = nn_param
        self.Q = q_function


    def sample(self, state, format="torch"):

        state = torch.Tensor(state).to(self.nn_params.device)
        self.batch_size = state.size()[0]

        q_values = self.Q.get_value(state, format="torch")
        actions = q_values.max(1)[1]

        sample_hot_vec = torch.tensor([[0.0 for i in range(self.nn_params.action_dim)]
                                       for j in range(self.batch_size)]).to(self.nn_params.device)

        for i in range(self.batch_size):
            sample_hot_vec[i][actions[i]] = 1

        if format == "torch":
            return sample_hot_vec
        elif format == "numpy":
            return sample_hot_vec.cpu().detach().numpy()


    def get_probabilities(self, state, format="torch"):
        probabilities = self.sample(state)
        if format == "torch":
            return probabilities
        elif format == "numpy":
            return probabilities.cpu().detach().numpy()

    def get_probability(self, state, action_no, format="torch"):
        probabilities = self.sample(state)
        prob = torch.reshape(probabilities[:, action_no], shape=(self.batch_size, 1))

        if format == "torch":
            return prob
        else:
            return prob.cpu().detach().numpy()

    def get_log_probability(self, state, action_no, format="torch"):
        if format == "torch":
            return torch.log(1e-8 + self.get_probability(state, action_no, format="torch")).to(self.nn_params.device)
        elif format == "numpy":
            return np.log(1e-8 + self.get_probability(state, action_no, format="numpy"))
