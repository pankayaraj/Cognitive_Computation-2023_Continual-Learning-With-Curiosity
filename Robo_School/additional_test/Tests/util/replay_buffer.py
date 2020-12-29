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
        self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask =  [], [], [], [], [], []

    def push(self, state, action, action_mean, reward, next_state, done_mask):
        if len(self.state) < self.capacity:
            self.state.append(None)
            self.action.append(None)
            self.action_mean.append(None)
            self.reward.append(None)
            self.next_state.append(None)
            self.done_mask.append(None)

            self.no_data += 1


        self.state[self.position] = state
        self.action[self.position] = action
        self.action_mean[self.position] = action_mean
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.done_mask[self.position] = done_mask

        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):

        indices = self.get_sample_indices(batch_size)

        state = np.take(np.array(self.state), indices, axis=0)
        action = np.take(np.array(self.action), indices, axis=0)
        action_mean = np.take(np.array(self.action_mean), indices, axis=0)
        reward = np.take(np.array(self.reward), indices, axis=0)
        next_state = np.take(np.array(self.next_state), indices, axis=0)
        done_mask = np.take(np.array(self.done_mask), indices, axis=0)

        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask)

    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask = [], [], [], [], [], []
        for i in indices:
            state.append(self.state[i])
            action.append(self.action[i])
            action_mean.append(self.action_mean[i])
            reward.append(self.reward[i])
            next_state.append(self.next_state[i])
            done_mask.append(self.done_mask[i])
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
        if len(self.state) < self.capacity:
            indices = np.random.choice(len(self.state), batch_size)
        else:
            indices = np.random.choice(self.capacity, batch_size)
        return indices


    def __len__(self):
        return self.no_data