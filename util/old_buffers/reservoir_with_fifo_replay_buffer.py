import numpy as np
from util.replay_buffer import Replay_Memory
from util.reservoir_replay_buffer import Reservoir_Replay_Memory
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask):
        #expects as list of items for each initalization variable
        self.state = np.array(state)
        self.action = np.array(action)
        self.action_mean = np.array(action_mean)
        self.reward = np.array(reward)
        self.next_state = np.array(next_state)
        self.done_mask = np.array(done_mask)

    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask]

class Reservoir_with_FIFO_Replay_Buffer():

    def __init__(self, capacity=10000, fifo_fac = 0.5):
        assert (fifo_fac > 0) and (fifo_fac < 1)
        self.fifo_frac = fifo_fac
        self.fifo_capacity = int(capacity*fifo_fac)
        self.reservior_capacity = capacity - self.fifo_capacity

        self.tiebreaker = count()

        self.fifo_buffer = Replay_Memory(capacity=self.fifo_capacity)
        self.reservior_buffer = Reservoir_Replay_Memory(capacity=self.reservior_capacity)

    def push(self, state, action, action_mean, reward, next_state, done_mask):
        ran = random.uniform(0,1)
        t = next(self.tiebreaker)
        if ran < self.fifo_frac:
            self.fifo_buffer.push(state, action, action_mean, reward, next_state, done_mask)
        else:
            self.reservior_buffer.push(state, action, action_mean, reward, next_state, done_mask, tie_breaker=t)

    def sample(self, batch_size):
        fifo_indices, reservior_indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask = self.encode_sample(fifo_indices, reservior_indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask)


    def encode_sample(self, fifo_indices, reservior_indices):

        state, action, action_mean, reward, next_state, done_mask = [], [], [], [], [], []
        s1, a1, a_m1, r1, n_s1, d1 = self.fifo_buffer.encode_sample(fifo_indices)
        s2, a2, a_m2, r2, n_s2, d2 = self.reservior_buffer.encode_sample(reservior_indices)

        state = state + s1 + s2
        action = action + a1 + a2
        action_mean = action_mean + a_m1 + a_m2
        reward = reward + r1 + r2
        next_state = next_state + n_s1 + n_s2
        done_mask = done_mask + d1 + d2

        return state, action, action_mean, reward, next_state, done_mask

    def get_sample_indices(self, batch_size):
        fifo_batch_size = int(batch_size * self.fifo_frac)
        reservior_batch_size = batch_size - fifo_batch_size

        fifo_indices = self.fifo_buffer.get_sample_indices(fifo_batch_size)
        reservior_indices = self.reservior_buffer.get_sample_indices(reservior_batch_size)

        return fifo_indices, reservior_indices


    def __len__(self):
        return len(self.fifo_buffer) + len(self.reservior_buffer)