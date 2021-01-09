import torch


class Hopper():
    def __init__(self):

        self.batch_size = 50000
        self.fifo_frac = 0.34
        self.avg_len = 400
        self.measre_reset_after_threshold = 28000
        self.measure_decrement = 1e-5
        self.snr_factor = 4.5

        self.change_varaiable_at = [1, 100000, 150000, 350000]
        self.change_varaiable = [0.75, 1.75, 2.75, 3.75]
        self.buffer_size = 50000
        self.fifo_frac = 0.34

        self.max_steps = 1000

class Hopper_avg_cur():

    def __init__(self):

        self.avg_len = 400
        self.measre_reset_after_threshold = 28000
        self.measure_decrement = 1e-5
        self.snr_factor = 5.5

        self.change_varaiable_at = [1, 100000, 150000, 350000]
        self.change_varaiable = [0.75, 1.75, 2.75, 3.75]
        self.buffer_size = 50000
        self.fifo_frac = 0.34

        self.max_steps = 1000



class Pendulum_avg_cur():

    def __init__(self):

        self.avg_len = 200
        self.measre_reset_after_threshold = 2400
        self.measure_decrement = 1e-4
        self.snr_factor = 5.5

        self.change_varaiable_at = [1, 30000, 60000, 120000, 200000]
        self.change_varaiable = [1.0, 1.2, 1.4, 1.6, 1.8]
        self.no_steps = 250000
        self.buffer_size = 8000
        self.fifo_frac = 0.34

        self.max_steps = 200