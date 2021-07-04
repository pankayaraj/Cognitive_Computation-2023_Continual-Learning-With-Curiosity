import numpy as np
import heapq
import random
from itertools import count

class Transition_tuple():

    def __init__(self, state, action, action_mean, reward, next_state, done_mask, t, initial_state, time_step):
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

    def get_all_attributes(self):
        return [self.state, self.action,  self.action_mean, self.reward, self.next_state, self.done_mask, self.t, self.initial_state, self.time_step]


class Custom_Res_TR():

    def __init__(self, capacity=10000, avg_len_snr=60, repetition_threshold=30000,
                 snr_factor=3, change_at = [100000, 350000]):
        self.capacity = capacity
        self.storage = [[]]
        self.residual_buffer = []
        self.tiebreaker = count()

        self.current_index = 0
        self.no_tasks = 1
        self.individual_buffer_capacity = capacity

        # task seperation parameters
        self.delta = 1e-10  # to avoid zero division

        self.avg_len_snr = avg_len_snr
        self.snr_factor = snr_factor
        self.last_spike_since = 0
        self.repetition_threshold = repetition_threshold

        self.curisoity_time_frame = [0 for i in range(avg_len_snr)]
        self.time = 0

        self.change_at = change_at
        self.change_count = 0

        # debug stuff
        self.PUSH = []
        self.SNR = []
        self.MEAN = []
        self.MEASURE = []
        self.BOOL = []
        self.max = 0

        self.task_seperation_initiated = False

        self.task_change_at_current_time_step = False


        self.split_sizes = [0]


    def task_change(self):
        self.task_change_at_current_time_step = True  # this flag needs to be reset to False by the algorithm that uses it

    # based on the decision made by the overarching algorithm it should make the decision on weather to partition the buffer or not using
    # these two subsequent functions

    def partition_buffer(self):

        l = []
        for b in self.storage:
            l.append(len(b))
        l.append(len(self.residual_buffer))
        print("task_change")
        print(self.time, l)

        self.current_index += 1
        self.no_tasks += 1
        self.individual_buffer_capacity = self.capacity // self.no_tasks

        x = self.capacity // (self.no_tasks * (self.no_tasks - 1))

        self.residual_buffer = []

        for (i, buff) in enumerate(self.storage):
            x_new = min(x, len(buff) - self.individual_buffer_capacity)
            print("x_new " + str(x_new))
            if len(buff) > self.individual_buffer_capacity:
                self.storage[i] = buff[x_new:]
                self.residual_buffer += buff[:x_new - 1]

        self.storage.append([])

    def consolidate_on_last_partition(self, buffer):
        print("consolidating reservoir buffer")
        self.external_buffer = buffer

        for d in self.external_buffer.storage:
            state, action, action_mean, reward, next_state, done_mask, tiebreaker, initial_state, time_step = d[2]
            self.push(state, action, action_mean, reward, next_state, done_mask, None, initial_state, time_step)

        print("reservoir buffer consolidated")


    def check_for_task_change(self):

        self.time = next(self.tiebreaker)  # both tiebreaker and timing is solved

        if self.change_count < len(self.change_at):
            if self.time == self.change_at[self.change_count]:
                self.change_count += 1
                self.task_change()

    def push(self, state, action, action_mean, reward,  next_state, done_mask, tiebreaker, initial_state, time_step):
        self.check_for_task_change()
        data = (state, action, action_mean, reward, next_state, done_mask, tiebreaker, initial_state, time_step)
        priority = random.uniform(0, 1)

        if tiebreaker == None:
            tiebreaker = self.time

        d = (priority, tiebreaker, data)

        if len(self.storage[self.current_index]) < self.individual_buffer_capacity:
            heapq.heappush(self.storage[self.current_index], d)
            pushed = True
        elif priority > self.storage[self.current_index][0][0]:
            heapq.heapreplace(self.storage[self.current_index], d)
            pushed = True
        else:
            pushed = False

        if pushed == True:
            if len(self.residual_buffer) != 0:
                self.residual_buffer.pop(0)

        return pushed

    def get_total_buffer_data(self):
        S = []
        for buff in self.storage:
            S += buff

        S += self.residual_buffer
        return S

    def sample(self, batch_size):
        indices = self.get_sample_indices(batch_size)
        state, action, action_mean, reward, next_state, done_mask, t_array, initial_state, time_step = self.encode_sample(
            indices=indices)
        return Transition_tuple(state, action, action_mean, reward, next_state, done_mask, t_array, initial_state, time_step)


    def encode_sample(self, indices):
        state, action, action_mean, reward, next_state, done_mask, t_array, initial_state, time_step = [], [], [], [], [], [], [], [], []
        for (j,idxs) in enumerate(indices):

            for i in idxs:
                if j == 0:
                    data = self.residual_buffer[i][2]
                else:

                    data = self.storage[j-1][i][2]
                s, a, a_m, r, n_s, d, t, i_s, t_s = data
                state.append(s)
                action.append(a)
                action_mean.append(a_m)
                reward.append(r)
                next_state.append(n_s)
                done_mask.append(d)
                t_array.append(t)
                initial_state.append(i_s)
                time_step.append(t_s)

        return state, action, action_mean, reward, next_state, done_mask, t_array, initial_state, time_step

    def get_sample_indices(self, batch_size):
        prop = self.get_proportion()
        batch_sizes = []
        temp = 0
        #for i in range(len(self.storage)-1):
        for i in range(len(self.storage)):
            temp += int(batch_size*prop[i])
            batch_sizes.append(int(batch_size*prop[i]))
        batch_sizes.append(batch_size-temp)

        indices = []
        for (i,buff) in enumerate(self.storage):
            if len(buff) < self.individual_buffer_capacity:
                indices.append(np.random.choice(len(buff), batch_sizes[i]))
            else:
                indices.append(np.random.choice(self.individual_buffer_capacity, batch_sizes[i]))

        # for residual buffer
        buff = self.residual_buffer
        if len(buff) != 0:
            indices.insert(0, np.random.choice(len(buff), batch_sizes[-1]))
        else:
            indices.insert(0, np.array([]))

        self.split_sizes = [len(inx) for inx in indices]
        self.debug = [prop, batch_size, self.split_sizes, batch_sizes, len(self.residual_buffer)]


        return indices



    def get_proportion(self):
        size = self.__len__()
        if size == 0:
            return [1.0]
        prop = []
        for buff in self.storage:
            prop.append(len(buff)/size)

        prop.append(len(self.residual_buffer)/size)

        return prop


    def __len__(self):
        l = 0
        for buff in self.storage:
            l += len(buff)

        l += len(self.residual_buffer)
        return l