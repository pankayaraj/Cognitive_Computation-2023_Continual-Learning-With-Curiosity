import numpy as np
import heapq
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask, t, initial_state, time_step, mean, std):

        #t : global, this is used to see whihc sample belong to which task at debuggung
        #time_step : episoidal, this is used for log ration calculation


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

class Reservoir_Replay_Memory_TR_P():

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.storage = []
        self.tiebreaker = count()

    def push(self, state, action, action_mean, reward, next_state, done_mask, tie_breaker, initial_state, time_step, mean, std ):
        data = (state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std)
        priority = random.uniform(0, 1)
        self.p = priority

        if tie_breaker == None:
            tie_breaker = next(self.tiebreaker)

        d = (priority, tie_breaker, data)

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
        state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std =  self.encode_sample(indices=indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask, None, initial_state,  time_step, mean, std)


    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std = [], [], [], [], [], [], [], [], [], []
        for i in indices:
            data = self.storage[i][2]
            s, a, a_m, r, n_s, d, i_s, t_s, m, st = data

            state.append(s)
            action.append(a)
            action_mean.append(a_m)
            reward.append(r)
            next_state.append(n_s)
            done_mask.append(d)
            initial_state.append(i_s)
            time_step.append(t_s)
            mean.append(m)
            std.append(st)


        return state, action, action_mean, reward, next_state, done_mask, initial_state, time_step, mean, std

    def get_sample_indices(self, batch_size):
        if len(self.storage) < self.capacity:
            indices = np.random.choice(len(self.storage), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)

        return indices

    def __len__(self):
        return len(self.storage)