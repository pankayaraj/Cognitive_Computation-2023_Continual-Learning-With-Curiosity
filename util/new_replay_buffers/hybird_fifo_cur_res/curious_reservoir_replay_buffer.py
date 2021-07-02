import numpy as np
import heapq
import random
from itertools import count
import torch
class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, curiosity, next_state, done_mask, t):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.curiosity = np.array(curiosity)
        self.next_state = np.array(next_state)
        self.done_mask = np.array(done_mask)
        self.t = np.array(t)

    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.curiosity, self.next_state, self.done_mask, self.t]


class Curious_Reservoir():

    def __init__(self, capacity=10000, ):
        self.capacity = capacity
        self.storage = [[]]
        self.residual_buffer = []
        self.tiebreaker = count()

        self.current_index = 0
        self.no_tasks = 1
        self.individual_buffer_capacity = capacity

        self.time = 0

        self.split_sizes = [0]


    def push(self, state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker):

        self.time = next(self.tiebreaker)

        data = (state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker)
        if str(type(curiosity)) != torch.Tensor:
            priority = curiosity
        else:
            priority = curiosity.item()


        if tiebreaker == None:
            tiebreaker = self.time

        d = (priority, tiebreaker, data)
        old_data = None

        if len(self.storage[self.current_index]) < self.individual_buffer_capacity:
            heapq.heappush(self.storage[self.current_index], d)
            pushed = True
            old_data = None
        elif priority > self.storage[self.current_index][0][0]:
            old_data = heapq.heapreplace(self.storage[self.current_index], d)
            pushed = True
        else:
            pushed = False

        if pushed == True:
            if len(self.residual_buffer) != 0:
                self.residual_buffer.pop(0)

        if pushed == False:
            return pushed, data
        else:
            return pushed, old_data

    def get_total_buffer_data(self):
        S = []
        for buff in self.storage:
            S += buff

        S += self.residual_buffer
        return S

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = self.encode_sample(
            indices=indices)
        return Transition_tuple(state, action, action_mean, reward, curiosity, next_state, done_mask, t_array)


    def encode_sample(self, indices):
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = [], [], [], [], [], [], [], []
        for (j,idxs) in enumerate(indices):
            for i in idxs:
                if j == 0:
                    data = self.residual_buffer[i][2]
                else:
                    data = self.storage[j-1][i][2]
                s, a, a_m, r, c, n_s, d, t = data
                state.append(s)
                action.append(a)
                action_mean.append(a_m)
                reward.append(r)
                curiosity.append(c)
                next_state.append(n_s)
                done_mask.append(d)
                t_array.append(t)

        return state, action, action_mean, reward, curiosity, next_state, done_mask, t_array

    def get_sample_indices(self, batch_size):
        prop = self.get_proportion()


        batch_sizes = []
        temp = 0
        #for i in range(len(self.storage)-1):
        for i in range(len(self.storage)):
            temp += int(batch_size*prop[i])
            batch_sizes.append(int(batch_size*prop[i]))
        batch_sizes.append(batch_size-temp) #this for residual buffer

        indices = []

        for (i,buff) in enumerate(self.storage):

            if len(buff) < self.individual_buffer_capacity:
                indices.append(np.random.choice(len(buff), batch_sizes[i]))
            else:
                indices.append(np.random.choice(self.individual_buffer_capacity, batch_sizes[i]))



        #for residual buffer
        buff = self.residual_buffer
        if len(buff) != 0:
            indices.insert(0, np.random.choice(len(buff), batch_sizes[-1]))
        else:
            indices.insert(0, np.array([]))


        #indices at which we can beform task split for IRM
        #(including the residual buffer, so u may ignore it)
        """
        self.split_indices = []
        curr_index = 0
        for i in range(len(self.storage)):
            curr_index = curr_index + len(indices[i])
            self.split_indices.append(curr_index)
        """
        self.split_sizes = [len(inx) for inx in indices]
        self.debug = [prop, batch_size, self.split_sizes, batch_sizes, len(self.residual_buffer) ]

        return indices

    def get_all_buff_sizes(self):
        buff_size = [len(buff) for buff in self.storage]
        return buff_size

    def get_proportion(self):
        size = self.__len__()
        if size == 0:
            return [1.0]
        prop = []
        for buff in self.storage:
            prop.append(len(buff)/size)

        prop.append(len(self.residual_buffer)/size)

        return prop


    def __len__(self):
        l = 0
        for buff in self.storage:
            l += len(buff)

        l += len(self.residual_buffer)
        return l