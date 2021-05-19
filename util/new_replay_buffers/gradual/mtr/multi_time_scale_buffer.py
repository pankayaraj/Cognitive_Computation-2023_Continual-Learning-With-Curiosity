import numpy as np
from collections import deque
from util.new_replay_buffers.gradual.mtr.replay_buffer_mtr import Replay_Memory_MTR

import random

def proportional(nsample, buf_sizes):
    T = np.sum(buf_sizes)
    S = nsample
    sample_sizes = [0 for i in range(len(buf_sizes))]
    for i in range(len(buf_sizes)):
        if S < 1:
            break
        sample_sizes[i] = int(round(S * buf_sizes[i] / T))
        T -= buf_sizes[i]
        S -= sample_sizes[i]
    assert sum(sample_sizes) == nsample, str(sum(sample_sizes))+" and "+str(nsample)
    return sample_sizes


class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask, t):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.done_mask = np.array(done_mask)
        self.t = np.array(t)

    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask, self.t]

class Multi_time_Scale_Buffer():

    def __init__(self, capacity, no_buffers, beta= 0.5, no_waste = True):

        self.no_buffers = no_buffers
        self.max_size_per_buffer = capacity//no_buffers
        self.max_size = no_buffers*self.max_size_per_buffer

        self.beta = beta
        self.no_waste = no_waste
        self.count = 0

        self.buffers = []
        for i in range(self.no_buffers):
            self.buffers.append(Replay_Memory_MTR(self.max_size_per_buffer))
        if self.no_waste:
            self.overflow_buffer = deque(maxlen=self.max_size)



    def __len__(self):
        total_len = 0
        for buff in self.buffers:
            total_len += len(buff)

        if self.no_waste:
            total_len += len(self.overflow_buffer)
        return total_len

    def buff_size(self):
        return self.max_size

    def push(self, state, action, action_mean, reward, next_state, done_mask):
        self.count += 1
        data = (state, action, action_mean, reward, next_state, done_mask, self.count)

        popped_data = self.buffers[0].push(state, action, action_mean, reward, next_state, done_mask, self.count)

        for i in range(1, self.no_buffers):
            if popped_data == None:
                break
            else:
                if random.uniform(0, 1) < self.beta:
                    popped_data = self.buffers[i].push(*popped_data)
                elif self.no_waste:
                    self.overflow_buffer.appendleft(popped_data)
                    break
                else:
                    break


        if self.no_waste and (self.count > self.max_size) and (len(self.overflow_buffer) != 0):
            self.overflow_buffer.pop()






    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask, time = [], [], [], [], [], [], []

        if self.no_waste:
            assert len(indices) == self.no_buffers + 1
        else:
            assert len(indices) == self.no_buffers

        for buff_i in range(len(indices)):
            for i in indices[buff_i]:
                if buff_i == 0 and self.no_waste:
                    data  = self.overflow_buffer[i]
                else:
                    data = self.buffers[buff_i -1].storage[i]

                s, a, a_m, r, n_s, d, t = data
                state.append(s)
                action.append(a)
                action_mean.append(a_m)
                reward.append(r)
                next_state.append(n_s)
                done_mask.append(d)
                time.append(t)

        return state, action, action_mean, reward, next_state, done_mask, time

    def get_sample_indices(self, batch_size):
        buff_len = [len(buff) for buff in self.buffers]

        if self.no_waste:
            buff_len.insert(0, len(self.overflow_buffer))
        #print(len(self.overflow_buffer))
        #print(buff_len)
        buff_batch_size = proportional(batch_size, buff_len)
        indices = [np.random.choice(buff_len[i], buff_batch_size[i]) for i in range(len(buff_len))]

        return indices

    def get_all_buff_sizes(self):
        buff_len = [len(buff) for buff in self.buffers]

        if self.no_waste:
            buff_len.insert(0, len(self.overflow_buffer))

        return buff_len

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask, time = self.encode_sample(indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask, time)


