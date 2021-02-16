import torch
class Pendulum_main():
    def __init__(self):

        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 600
        self.measre_reset_after_threshold = 8000
        self.snr_factor = 1.5

        self.cur_rew_inverse_ratio = 0.0
        self.automatic_tuning = True

        self.change_varaiable_at = [1, 20000, 120000]
        self.change_varaiable = [1.0, 1.4, 1.8]
        self.buffer_size = 20000

        self.max_steps = 200

class Hopper_main():
    def __init__(self):

        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 500
        self.measre_reset_after_threshold = 30000
        self.snr_factor = 2.5

        self.cur_rew_inverse_ratio = 0.0
        self.automatic_tuning = True

        self.change_varaiable_at = [1, 50000, 350000]
        self.change_varaiable = [0.75, 4.75, 8.75]
        self.buffer_size = 50000

        self.max_steps = 1000


class Walker2D_main():  #[1.4, 7.4, 13.4] , [0, 250k, 350k, 400k]
    def __init__(self):

        self.fifo_frac = 0.05
        self.curisoity_buff_frac = 0.34
        self.avg_len = 2000
        self.measre_reset_after_threshold = 30000
        self.snr_factor = 0.2

        self.cur_rew_inverse_ratio = 0.05
        self.automatic_tuning = False
        self.alpha = 0.2


        self.change_varaiable_at = [1, 250000, 350000]
        self.change_varaiable = [1.4, 7.4, 13.4]
        self.buffer_size = 100000

        self.max_steps = 1000





