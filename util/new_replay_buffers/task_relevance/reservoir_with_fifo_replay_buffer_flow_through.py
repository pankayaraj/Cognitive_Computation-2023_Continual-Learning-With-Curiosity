import numpy as np
from util.new_replay_buffers.task_relevance.replay_buffer import Replay_Memory_TR
from util.new_replay_buffers.task_relevance.reservoir_replay_buffer import Reservoir_Replay_Memory_TR
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask, t, time_step):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.done_mask = np.array(done_mask)
        self.t = np.array(t)
        self.time_step = time_step

    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask, self.t, self.time_step ]

class Half_Reservoir_with_FIFO_Flow_Through_Replay_Buffer_TR():

    def __init__(self, capacity=10000, fifo_fac = 0.5, avg_len_snr=60, measre_reset_after_threshold=40000,
                 measure_decrement=1e-4, snr_factor=3,):
        assert (fifo_fac > 0) and (fifo_fac < 1)
        self.fifo_frac = fifo_fac
        self.fifo_capacity = int(capacity*fifo_fac)
        self.reservior_capacity = capacity - self.fifo_capacity

        self.fifo_buffer = Replay_Memory_TR(capacity=self.fifo_capacity)
        self.reservior_buffer = Reservoir_Replay_Memory_TR(capacity=self.reservior_capacity)
        self.t = 0


    def push(self, state, action, action_mean, reward, next_state, done_mask, time_step):
        self.t += 1

        old_data = self.fifo_buffer.push(state, action, action_mean, reward, next_state, done_mask, time_step)
        if old_data != None:
            state, action, action_mean, reward, next_state, done_mask, time_step = old_data
            self.reservior_buffer.push(state, action, action_mean, reward, next_state, done_mask, None, time_step)

    def sample(self, batch_size):
        fifo_indices, reservoir_indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask, time_step = self.encode_sample(fifo_indices, reservoir_indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask, None, time_step)


    def encode_sample(self, fifo_indices, reservior_indices):

        state, action, action_mean, reward, next_state, done_mask, time_step = [], [], [], [], [], [], []
        s1, a1, a_m1, r1, n_s1, d1, ts1 = self.fifo_buffer.encode_sample(fifo_indices)
        s2, a2, a_m2, r2, n_s2, d2, ts2 = self.reservior_buffer.encode_sample(reservior_indices)

        state = state + s1 + s2
        action = action + a1 + a2
        action_mean = action_mean + a_m1 + a_m2
        reward = reward + r1 + r2
        next_state = next_state + n_s1 + n_s2
        done_mask = done_mask + d1 + d2
        time_step = ts1 + ts2
        return state, action, action_mean, reward, next_state, done_mask, time_step

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