import numpy as np


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

class Replay_Memory():

    def __init__(self, capacity=10000):
        self.no_data = 0
        self.position = 0
        self.capacity = capacity
        self.storage = []

    def push(self, state, action, action_mean, reward, next_state, done_mask):
        data = (state, action, action_mean, reward, next_state, done_mask)

        if len(self.storage) < self.capacity:
            self.storage.append(data)
            self.no_data += 1
            self.position = (self.position + 1) % self.capacity

            return

        old_data = self.storage[self.position]
        self.storage[self.position] = data
        self.position = (self.position + 1) % self.capacity

        #for MTR
        return old_data


    def sample(self, batch_size):

        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask = self.encode_sample(indices)

        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask)

    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask = [], [], [], [], [], []
        for i in indices:
            state.append(self.storage[i][0])
            action.append(self.storage[i][1])
            action_mean.append(self.storage[i][2])
            reward.append(self.storage[i][3])
            next_state.append(self.storage[i][4])
            done_mask.append(self.storage[i][5])
        return state, action, action_mean, reward, next_state, done_mask

    def iterate_through(self):
        all_data = self.sample(self.no_data)
        all_attributes = all_data.get_all_attributes()

        for i in range(self.no_data):
            t = []
            for j in range(len(all_attributes)):
                t.append(all_attributes[j][i])

            yield t

    def get_sample_indices(self, batch_size):
        if len(self.storage) < self.capacity:
            indices = np.random.choice(len(self.storage), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)
        return indices


    def __len__(self):
        return self.no_data