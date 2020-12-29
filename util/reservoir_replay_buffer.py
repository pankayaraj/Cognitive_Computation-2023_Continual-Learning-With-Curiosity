import numpy as np
import heapq
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

class Reservoir_Replay_Memory():

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.storage = []
        self.tiebreaker = count()

    def push(self, state, action, action_mean, reward, next_state, done_mask):
        data = (state, action, action_mean, reward, next_state, done_mask)
        priority = random.uniform(0, 1)


        d = (priority, next(self.tiebreaker), data)

        if len(self.storage) < self.capacity:
            heapq.heappush(self.storage, d)
        elif priority > self.storage[0][0]:
            heapq.heapreplace(self.storage, d)

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask =  self.encode_sample(indices=indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask)



    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask = [], [], [], [], [], []
        for i in indices:
            data = self.storage[i][2]
            s, a, a_m, r, n_s, d = data

            state.append(s)
            action.append(a)
            action_mean.append(a_m)
            reward.append(r)
            next_state.append(n_s)
            done_mask.append(d)
        return state, action, action_mean, reward, next_state, done_mask

    def get_sample_indices(self, batch_size):
        if len(self.storage) < self.capacity:
            indices = np.random.choice(len(self.storage), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)

        return indices

    def __len__(self):
        return len(self.storage)