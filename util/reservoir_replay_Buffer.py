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

class Reservoir_Replay_Memory():

    def __init__(self):
        pass