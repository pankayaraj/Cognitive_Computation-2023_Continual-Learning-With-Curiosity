import torch
class PendulumV0():
    def __init__(self):

        self.buff_size = 20000
        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 600
        self.measre_reset_after_threshold = 8000
        self.snr_factor = 1.5

        self.change_varaiable_at = [1, 20000, 120000]
        self.change_varaiable = [1.0, 1.4, 1.8]
        self.buffer_size = 20000

        self.max_steps = 200

class Hopper():
    def __init__(self):

        self.buff_size = 50000
        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 500
        self.measre_reset_after_threshold = 30000
        self.snr_factor = 2.5

        self.change_varaiable_at = [1, 50000, 350000]
        self.change_varaiable = [0.75, 4.75, 8.75]
        self.buffer_size = 50000

        self.max_steps = 1000


class Walker2D_1():  #[0.4, 0.90, 1.40] , [0, 50k, 350k, 400k]
    def __init__(self):
        self.buff_size = 50000
        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 600
        self.measre_reset_after_threshold = 30000
        self.snr_factor = 1.5

        self.change_varaiable_at = [1, 50000, 350000]
        self.change_varaiable = [0.40, 0.90, 1.40]
        self.buffer_size = 50000

        self.max_steps = 1000



class Walker2D_2():  #[0.4, 1.15, 1.90] , [0, 50k, 350k, 400k]
    def __init__(self):
        self.batch_size = 50000
        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 600
        self.measre_reset_after_threshold = 30000
        self.snr_factor = 1.5

        self.change_varaiable_at = [1, 50000, 350000]
        self.change_varaiable = [0.40, 1.15, 1.90]
        self.buffer_size = 50000

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