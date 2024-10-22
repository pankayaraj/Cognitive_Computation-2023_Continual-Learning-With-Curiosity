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


class Reservoir_Task_Seperation_Replay_Memory_Gradual():

    def __init__(self, capacity=10000, avg_len_snr=60, repetition_threshold=30000,
                 snr_factor=3, snr_factor_secondary=3.0, priority = "uniform" ):
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

        self.priority = priority

        self.avg_len_snr_sec = avg_len_snr
        self.snr_factor_secondary = snr_factor_secondary
        self.last_spike_since_sec = 0
        self.repetition_threshold_sec = repetition_threshold

        self.curisoity_time_frame = [0 for i in range(avg_len_snr)]
        self.curisoity_time_frame_sec = [0 for i in range(avg_len_snr)]
        self.time = 0

        # debug stuff
        self.PUSH = []
        self.SNR = []
        self.MEAN = []
        self.STD = []
        self.MEASURE = []
        self.BOOL = []
        self.max = 0

        self.PUSH_sec = []
        self.SNR_sec = []
        self.MEAN_sec = []
        self.MEASURE_sec = []
        self.BOOL_sec = []
        self.max_sec = 0


        self.t_c = False
        self.t_c_limit = 5000
        self.t_c_counter = 0

        self.task_seperation_initiated = False

        self.split_sizes = [0]

    def task_change(self):

        l = []
        for  b in self.storage:
            l.append(len(b))
        l.append(len(self.residual_buffer))
        print("task_change")
        print(self.time, l)

        self.current_index += 1
        self.no_tasks += 1
        self.individual_buffer_capacity = self.capacity//self.no_tasks

        x = self.capacity//(self.no_tasks*(self.no_tasks-1))

        self.residual_buffer = []

        for (i, buff) in enumerate(self.storage):
            x_new = min(x, len(buff)-self.individual_buffer_capacity)
            print("x_new " + str(x_new))
            if len(buff) > self.individual_buffer_capacity:
                self.storage[i] = buff[x_new:]
                self.residual_buffer += buff[:x_new-1]



        self.storage.append([])



    def check_for_task_change(self, curiosity):

        self.time = next(self.tiebreaker)  # both tiebreaker and timing is solved


        cur = curiosity

        if self.time < self.avg_len_snr:
            self.curisoity_time_frame[self.time] = cur

        else:
            self.curisoity_time_frame.pop(0)
            self.curisoity_time_frame.append(cur)

            mean = np.mean(self.curisoity_time_frame).item()
            std = np.std(self.curisoity_time_frame).item()
            SNR = mean / (std + self.delta)

            # setting the idling threshold
            if SNR < self.snr_factor * mean:

                self.BOOL.append(1.0)

                if self.last_spike_since > self.repetition_threshold:
                    self.task_change()

                self.last_spike_since = 0
            else:


                self.BOOL.append(0.0)
                self.last_spike_since += 1

            self.SNR.append(SNR)
            self.MEAN.append(mean)
            self.STD.append(std)

    def check_for_task_change_secondary(self, curiosity):
        #not needed as it happend in the function above
        #self.time = next(self.tiebreaker)  # both tiebreaker and timing is solved


        cur = curiosity

        if self.time < self.avg_len_snr_sec:
            self.curisoity_time_frame_sec[self.time] = cur

            if self.t_c_counter > self.t_c_limit:
                self.t_c = False
            else:
                self.t_c_counter += 1

        else:
            self.curisoity_time_frame_sec.pop(0)
            self.curisoity_time_frame_sec.append(cur)

            mean = np.mean(self.curisoity_time_frame_sec).item()
            std = np.std(self.curisoity_time_frame_sec).item()
            SNR = mean / (std + self.delta)

            # setting the idling threshold
            if SNR < self.snr_factor_secondary * mean:

                self.BOOL_sec.append(1.0)

                if self.last_spike_since_sec > self.repetition_threshold_sec:
                    #self.task_change()
                    print("secondary_chnage")
                    self.t_c = True
                    self.t_c_limit = 0

                else:
                    if self.t_c_counter > self.t_c_limit:
                        self.t_c = False
                    else:
                        self.t_c_counter += 1

                self.last_spike_since_sec = 0
            else:

                if self.t_c_counter > self.t_c_limit:
                    self.t_c = False
                else:
                    self.t_c_counter += 1


                self.BOOL_sec.append(0.0)
                self.last_spike_since_sec += 1

            self.SNR_sec.append(SNR)
            self.MEAN_sec.append(mean)


    def push(self, state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker):
        #maintain the order of function call
        #FOR BUFFER WITH TASK SEPERATION
        self.check_for_task_change(curiosity=curiosity)


        #FOR CURIOSITY RESERVOIR ALONE
        #self.time = next(self.tiebreaker)

        data = (state, action, action_mean, reward, curiosity, next_state, done_mask, tiebreaker)

        if self.priority == "uniform":
            priority = random.uniform(0, 1)
        elif self.priority == "curiosity":
            if str(type(curiosity)) != torch.Tensor:
                priority = curiosity
            else:
                priority = curiosity.item()


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
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = self.encode_sample(
            indices=indices)
        return Transition_tuple(state, action, action_mean, reward, curiosity, next_state, done_mask, t_array)


    def encode_sample(self, indices):
        state, action, action_mean, reward, curiosity, next_state, done_mask, t_array = [], [], [], [], [], [], [], []
        for (j,idxs) in enumerate(indices):
            for i in idxs:
                if j == 0:
                    data = self.residual_buffer[i][2]
                else:
                    data = self.storage[j-1][i][2]
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
        prop = self.get_proportion()


        batch_sizes = []
        temp = 0
        #for i in range(len(self.storage)-1):
        for i in range(len(self.storage)):
            temp += int(batch_size*prop[i])
            batch_sizes.append(int(batch_size*prop[i]))
        batch_sizes.append(batch_size-temp) #this for residual buffer

        indices = []

        for (i,buff) in enumerate(self.storage):

            if len(buff) < self.individual_buffer_capacity:
                indices.append(np.random.choice(len(buff), batch_sizes[i]))
            else:
                indices.append(np.random.choice(self.individual_buffer_capacity, batch_sizes[i]))



        #for residual buffer
        buff = self.residual_buffer
        if len(buff) != 0:
            indices.insert(0, np.random.choice(len(buff), batch_sizes[-1]))
        else:
            indices.insert(0, np.array([]))


        #indices at which we can beform task split for IRM
        #(including the residual buffer, so u may ignore it)
        """
        self.split_indices = []
        curr_index = 0
        for i in range(len(self.storage)):
            curr_index = curr_index + len(indices[i])
            self.split_indices.append(curr_index)
        """
        self.split_sizes = [len(inx) for inx in indices]
        self.debug = [prop, batch_size, self.split_sizes, batch_sizes, len(self.residual_buffer) ]

        return indices

    def get_all_buff_sizes(self):
        buff_size = [len(buff) for buff in self.storage]
        return buff_size

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