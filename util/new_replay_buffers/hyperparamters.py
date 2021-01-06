import torch


class Hopper():
    def __init__(self):

        self.avg_len = 400
        self.measre_reset_after_threshold = 28000
        self.measure_decrement = 1e-5
        self.snr_factor = 4.5

