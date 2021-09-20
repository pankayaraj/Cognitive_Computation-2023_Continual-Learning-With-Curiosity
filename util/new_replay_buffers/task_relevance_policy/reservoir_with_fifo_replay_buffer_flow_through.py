import numpy as np
from util.new_replay_buffers.task_relevance_policy.replay_buffer import Replay_Memory_TR_P
from util.new_replay_buffers.task_relevance_policy.reservoir_replay_buffer import Reservoir_Replay_Memory_TR_P
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask, t, initial_state, time_step, mean, std):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.done_mask = np.array(done_mask)
        self.t = np.array(t)
        self.initial_state = np.array(initial_state)
        self.time_step = np.array(time_step)
        self.mean = np.array(mean)
        self.std = np.array(std)


    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask, self.t, self.initial_state, self.time_step, self.mean, self.std]

class Half_Reservoir_with_FIFO_Flow_Through_Replay_Buffer_TR_P():

    def __init__(self, capacity=10000, fifo_fac = 0.5, avg_len_snr=60, measre_reset_after_threshold=40000,
                 measure_decrement=1e-4, snr_factor=3,):
        assert (fifo_fac > 0) and (fifo_fac < 1)
        self.fifo_frac = fifo_fac
        self.fifo_capacity = int(capacity*fifo_fac)
        self.reservior_capacity = capacity - self.fifo_capacity

        self.fifo_buffer = Replay_Memory_TR_P(capacity=self.fifo_capacity)
        self.reservior_buffer = Reservoir_Replay_Memory_TR_P(capacity=self.reservior_capacity)
        self.t = 0


    def push(self, state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std):
        self.t += 1

        old_data = self.fifo_buffer.push(state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std)
        if old_data != None:
            state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std = old_data
            self.reservior_buffer.push(state, action, action_mean, reward, next_state, done_mask, None, initial_state, time_step, mean, std)

    def sample(self, batch_size):
        fifo_indices, reservoir_indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std = self.encode_sample(fifo_indices, reservoir_indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask, None, initial_state, time_step, mean, std)


    def encode_sample(self, fifo_indices, reservior_indices):

        state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std = [], [], [], [], [], [], [], [], [], []
        s1, a1, a_m1, r1, n_s1, d1, i_s1, ts1, m1, st1 = self.fifo_buffer.encode_sample(fifo_indices)
        s2, a2, a_m2, r2, n_s2, d2, i_s2, ts2, m2, st2 = self.reservior_buffer.encode_sample(reservior_indices)

        state = state + s1 + s2
        action = action + a1 + a2
        action_mean = action_mean + a_m1 + a_m2
        reward = reward + r1 + r2
        next_state = next_state + n_s1 + n_s2
        done_mask = done_mask + d1 + d2
        initial_state = i_s1 + i_s2
        time_step = ts1 + ts2
        mean = m1 + m2
        std = st1 + st2

        return state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std

    def get_sample_indices(self, batch_size):

        fifo_batch_size = int(batch_size * self.fifo_frac)
        reservoir_batch_size = batch_size - fifo_batch_size
        if len(self.reservior_buffer) < reservoir_batch_size:
            reservoir_batch_size = len(self.reservior_buffer)
            fifo_batch_size = batch_size - reservoir_batch_size

        fifo_indices = self.fifo_buffer.get_sample_indices(fifo_batch_size)
        reservoir_indices = self.reservior_buffer.get_sample_indices(reservoir_batch_size)

        return fifo_indices, reservoir_indices


    def __len__(self):
        return len(self.fifo_buffer) + len(self.reservior_buffer)