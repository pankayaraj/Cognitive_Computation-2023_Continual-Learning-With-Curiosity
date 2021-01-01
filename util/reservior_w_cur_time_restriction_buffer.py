import numpy as np
import random
import heapq
from itertools import count

def time_fun(x, slope, shift):
    return 1/(1+np.exp((slope*x - shift)))

class eligibility_trace():

    def __init__(self, lambda_v, r, slope=3, shift=5):

        self.E = 0
        self.lambda_v = lambda_v
        self.r = r
        self.slope = slope
        self.shift = shift

    def get_trace(self):
        return time_fun(self.E, slope=self.slope, shift=self.shift)

    def general_iterate(self):
        self.E = self.r*self.lambda_v*self.E

    def is_pushed(self):
        self.E += 1

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

class Reservoir_with_Cur_n_Time_Restirction_Replay_Memory():

    def __init__(self, capacity=10000, lambda_v=0.5, r=1, slope=3, shift=5):
        self.capacity = capacity
        self.storage = []
        self.tiebreaker = count()
        self.time_eligibilty_trace = eligibility_trace(lambda_v=lambda_v, r=r, slope=slope, shift=shift)
        self.t = 0

    def push(self, state, action, action_mean, reward, curiosity, next_state, done_mask, t=None):

        if t == None:
            t = self.t
            tiebreaker = next(self.tiebreaker)
        else:
            tiebreaker = t
        self.t += 1


        ran = random.uniform(0,1)
        self.time_eligibilty_trace.general_iterate()

        if ran < self.time_eligibilty_trace.get_trace():
            data = (state, action, action_mean, reward, curiosity, next_state, done_mask, t)
            priority = curiosity.item()

            d = (priority, tiebreaker, data)

            if len(self.storage) < self.capacity:
                heapq.heappush(self.storage, d)
                return True
            elif priority > self.storage[0][0]:
                heapq.heapreplace(self.storage, d)
                self.time_eligibilty_trace.is_pushed()
                return True
            else:
                return False
        else:
            return False

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array =  self.encode_sample(indices=indices)
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