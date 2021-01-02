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


class Reservoir_with_Cur_SNR_Replay_Memory():

    def __init__(self, capacity=10000, avg_len_snr=60, measre_reset_after_threshold=40000,
                 measure_decrement=1e-4, snr_factor=3, ):
        self.capacity = capacity
        self.storage = []
        self.tiebreaker = count() #this also keeps the count of the time step the algorithm is currently at


        #task seperation parameters
        self.delta = 1e-10 #to avoid zero division

        self.avg_len_snr = avg_len_snr
        self.snr_factor = snr_factor
        self.measure_threshold = 1.0
        self.last_push_since = 0
        self.measure_reset_after_threshold = measre_reset_after_threshold
        self.measure_decrement = measure_decrement

        self.curisoity_time_frame = [0 for i in range(avg_len_snr)]

        self.t = 0

        self.PUSH = []
        self.SNR = []
        self.MEAN = []
        self.MEASURE = []

    def push(self, state, action, action_mean, reward, curiosity, next_state, done_mask, global_time=None):
        time = next(self.tiebreaker) #both tiebreaker and timing is solved

        if global_time == None:
            self.t = time
        else:
            self.t = global_time

        cur = curiosity.item()

        if time < self.avg_len_snr:
            self.curisoity_time_frame[time] = cur
            pushed = self.internal_push(state,action,action_mean,reward,curiosity,next_state,done_mask,tiebreaker=self.t)

            return pushed
        else:
            self.curisoity_time_frame.pop(0)
            self.curisoity_time_frame.append(cur)

            mean = np.mean(self.curisoity_time_frame).item()
            std = np.std(self.curisoity_time_frame).item()
            SNR = mean/(std+self.delta)

            #setting the idling threshold
            if SNR < self.snr_factor*mean:
                self.last_push_since = 0
            else:
                self.last_push_since += 1

            #setting the measure threshold
            if self.last_push_since > self.measure_reset_after_threshold:
                self.measure_threshold = 1.0
            else:
                self.measure_threshold -= self.measure_decrement

            if self.measure_threshold > 0.5:
                pushed = self.internal_push(state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker=self.t)
            else:
                pushed = False

            #for debugging

            """
            if pushed:
                self.PUSH.append(1.0)
            else:
                self.PUSH.append(0.0)


            self.SNR.append(SNR)
            self.MEAN.append(mean)
            self.MEASURE.append(self.measure_threshold)
            """

            return pushed

    def internal_push(self, state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker):

        data = (state, action, action_mean, reward, curiosity, next_state, done_mask)
        priority = curiosity.item()

        d = (priority, tiebreaker, data)

        if len(self.storage) < self.capacity:
            heapq.heappush(self.storage, d)
            return True
        elif priority > self.storage[0][0]:
            heapq.heapreplace(self.storage, d)
            return True
        else:
            return False

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = self.encode_sample(indices=indices)
        return Transition_tuple(state, action, action_mean, reward, curiosity, next_state, done_mask, t_array)

    def encode_sample(self, indices):
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = [], [], [], [], [], [], [], []
        for i in indices:
            data = self.storage[i][2]
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
        if len(self.storage) < self.capacity:
            indices = np.random.choice(len(self.storage), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)

        return indices

    def __len__(self):
        return len(self.storage)