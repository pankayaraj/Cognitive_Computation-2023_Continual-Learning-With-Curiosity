import numpy as np
from util.new_replay_buffers.replay_buff_cur import Replay_Memory_Cur
from util.new_replay_buffers.hybird_fifo_cur_res.curious_reservoir_replay_buffer import Curious_Reservoir
from util.new_replay_buffers.hybird_fifo_cur_res.reservoir_replay_buffer_new import Reservoir_Replay_Memory_new
import random
from itertools import count

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

class Hybrid_fifo_cur_res_res():

    def __init__(self, capacity=10000, curisoity_buff_frac = 0.34, seperate_cur_buffer=True,
                 fifo_fac=0.05, cur_res_frac = 0.5):
        assert (fifo_fac > 0) and (fifo_fac < 1)
        assert (cur_res_frac > 0) and (cur_res_frac < 1)

        self.fifo_frac = fifo_fac
        self.cur_res_frac = cur_res_frac
        self.fifo_capacity = int(capacity*fifo_fac)
        self.cur_reservior_capacity = int(capacity*cur_res_frac)
        self.reservior_capacity  =  capacity - self.fifo_capacity - self.cur_reservior_capacity

        self.curiosity_buffer = Replay_Memory_Cur(capacity=int(capacity*curisoity_buff_frac))
        self.seperate_cur_buffer = seperate_cur_buffer

        self.fifo_buffer = Replay_Memory_Cur(capacity=self.fifo_capacity)
        self.reservior_buffer = Reservoir_Replay_Memory_new(capacity=self.reservior_capacity,)
        self.cur_reservior_buffer = Curious_Reservoir(capacity=self.cur_reservior_capacity)

        self.t = 0
        self.split_sizes  = [0]

    def push(self, state, action, action_mean, reward, curiosity, next_state, done_mask):
        self.t += 1

        old_data = self.fifo_buffer.push(state, action, action_mean, reward, curiosity, next_state, done_mask)
        if old_data != None:
            state, action, action_mean, reward, curiosity, next_state, done_mask = old_data
            pushed, old_data = self.cur_reservior_buffer.push(state, action, action_mean, reward, curiosity, next_state, done_mask, self.t)

            if old_data != None:
                state, action, action_mean, reward, curiosity, next_state, done_mask, time = old_data
                self.reservior_buffer.push(state, action, action_mean, reward, curiosity, next_state, done_mask, self.t)

        self.curiosity_buffer.push(state, action, action_mean, reward, curiosity, next_state, done_mask)

    def sample(self, batch_size):

        fifo_indices, cur_reservoir_indices, reservoir_indices,  = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, curiosity, next_state, done_mask = self.encode_sample(fifo_indices, cur_reservoir_indices, reservoir_indices)

        fifo_batch_size = int(batch_size * self.fifo_frac)

        #NEED TO ADD THE SAME FOR CUR_RESERVOIR_BUFFER
        #self.reservior_buffer.split_sizes.append(fifo_batch_size)
        #self.split_sizes = self.reservior_buffer.split_sizes

        return Transition_tuple(state, action, action_mean, reward, curiosity, next_state, done_mask,None)

    def sample_for_curiosity(self, batch_size):
        return self.curiosity_buffer.sample(batch_size)


    def encode_sample(self, fifo_indices, cur_reservior_indices, reservior_indices):


        state, action, action_mean, reward, curiosity, next_state, done_mask = [], [], [], [], [], [], []
        s1, a1, a_m1, r1, c1, n_s1, d1 = self.fifo_buffer.encode_sample(fifo_indices)
        s2, a2, a_m2, r2, c2, n_s2, d2, t2 = self.cur_reservior_buffer.encode_sample(cur_reservior_indices)
        s3, a3, a_m3, r3, c3, n_s3, d3, t3 = self.reservior_buffer.encode_sample(reservior_indices)

        #conveniently_added_fifo buffer data at the end to facilitate split in case if IRM
        state = state + s3 + s2 + s1
        action = action + a3 + a2 + a1
        action_mean = action_mean + a_m3 + a_m2 + a_m1
        reward = reward + r3 + r2 + r1
        curiosity = curiosity + c3 + c2 + c1
        next_state = next_state + n_s3 + n_s2 + n_s1
        done_mask = done_mask + d3 + d2 + d1

        return state, action, action_mean, reward, curiosity, next_state, done_mask

    def get_sample_indices(self, batch_size):

        fifo_batch_size = int(batch_size * self.fifo_frac)
        cur_reservoir_batch_size = int(batch_size*self.cur_res_frac)
        reservoir_batch_size = batch_size - fifo_batch_size - cur_reservoir_batch_size



        if len(self.cur_reservior_buffer) < cur_reservoir_batch_size:
            cur_reservoir_batch_size = len(self.reservior_buffer)
            reservoir_batch_size = 0
            fifo_batch_size = batch_size - cur_reservoir_batch_size

        elif len(self.reservior_buffer) < reservoir_batch_size:
            reservoir_batch_size = len(self.reservior_buffer)
            fifo_batch_size = int(batch_size * self.fifo_frac)
            cur_reservoir_batch_size = batch_size - reservoir_batch_size - fifo_batch_size


        fifo_indices = self.fifo_buffer.get_sample_indices(fifo_batch_size)
        cur_reservoir_indices = self.cur_reservior_buffer.get_sample_indices(cur_reservoir_batch_size)
        reservoir_indices = self.reservior_buffer.get_sample_indices(reservoir_batch_size)


        return fifo_indices, cur_reservoir_indices, reservoir_indices

    #def get_all_buff_sizes(self):
    #    res_cap = self.reservior_buffer.get_all_buff_sizes()
    #    buff_sizes = res_cap.insert(0, len(self.fifo_buffer))

    #    return buff_sizes

    def __len__(self):
        return len(self.fifo_buffer) + len(self.reservior_buffer) + len(self.cur_reservior_buffer)