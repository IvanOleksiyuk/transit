import numpy as np

class Linear():
    def __init__(self, start_tokens = 0, start_epoch = 0, end_epoch = 0, end_tokens=0):
        self.start_tokens = start_tokens
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.end_tokens = end_tokens
        
    def __call__(self, epoch):
        if epoch < self.start_epoch:
            return self.start_tokens 
        elif epoch > self.end_epoch:
            return self.end_tokens  
        else:
            return int((epoch - self.start_epoch) / (self.end_epoch - self.start_epoch) * (self.end_tokens - self.start_tokens) + self.start_tokens)
        