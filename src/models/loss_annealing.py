import numpy as np 

class SinAnnealing():
    def __init__(self, amplitude, period, start = None, end = None, min_val = 0):
        self.amplitude = amplitude
        self.start = start
        self.end = end
        self.period = period
        self.min_val = min_val
        
    def __call__(self, batch):
        if (self.start is not None) and batch < self.start:
            return self.min_val
        elif (self.end is not None) and batch > self.end:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (self.end - self.start)) ** 2
        else:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (batch - self.start)) ** 2

class SigmoidAnnealing():
    def __init__(self, amplitude = None, start = None, end = None, min_val = 0, steepness=5):
        
        self.start = start
        self.end = end
        self.min_val = min_val
        self.slope = 10/(end - start)
        self.steepness = steepness
        self.amplitude = amplitude
        self.middle = (start + end) / 2
        
    def __call__(self, batch):
        if (self.start is not None) and batch < self.start:
            return self.min_val
        elif (self.end is not None) and batch > self.end:
            return self.min_val + self.amplitude / (1 + np.exp(-self.slope * (self.end - self.middle)))
        else:
            return self.min_val + self.amplitude / (1 + np.exp(-self.slope * (batch - self.middle))) 
        
class ExponentialAnnealing():
    def __init__(self, start = None, end = None, start_val = 0, end_val=1):
        
        self.start = start
        self.end = end
        self.start_val = start_val
        self.end_val = end_val
        self.slope = (np.log(end_val)-np.log(start_val))/(end - start)
        
    def __call__(self, batch):
        if batch < self.start:
            return self.start_val 
        elif batch > self.end:
            return self.end_val  
        else:
            return np.exp(self.slope * (batch - self.start)+np.log(self.start_val))
        