import numpy as np

class Trajectory:
    def forward(self, point, t):
        raise NotImplementedError
    
    def inverse(self, point, t):
        raise NotImplementedError

class ConstantTrajectory(Trajectory):
    def __init__(self, n_vars=None):
        pass

    def forward(self, point, t):
        return point
    
    def inverse(self, point, t):
        return point

class LinearTrajectory(Trajectory):
    def __init__(self, n_vars, velocity="random"):
        if isinstance(velocity, str) and velocity == "random":
            self.velocity = np.random.uniform(-0.5, 0.5, size=n_vars)
        elif velocity is None:
            self.velocity = np.ones(n_vars)
        else:
            self.velocity = np.array(velocity)
            
    def forward(self, point, t):
        return point + self.velocity * t
    
    def inverse(self, point, t):
        return point - self.velocity * t

class QuadraticTrajectory(Trajectory):
    def __init__(self, n_vars, a=None, peak_t=None, lin=None):
        self.n_vars = n_vars
        def _init_param(p):
            if p is None: return np.random.uniform(-1, 1, size=n_vars)
            return np.array(p)
        
        self.a = _init_param(a)
        self.peak_t = _init_param(peak_t)
        self.lin = _init_param(lin)

    def forward(self, point, t):
        return point + self.lin * t + 0.5 * self.a * (t - self.peak_t)**2

    def inverse(self, point, t):
        return point - (self.lin * t + 0.5 * self.a * (t - self.peak_t)**2)