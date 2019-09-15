import numpy as np


class OUNoise:

    def __init__(self, mu: np.ndarray):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    @classmethod
    def from_action_space(cls, action_space):
        return cls(np.ones(action_space.shape))

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class NormalNoise:

    def __init__(self, mu: np.ndarray, sigma: np.ndarray=None, decay=1., min_sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.decay = decay
        self.min_sigma = min_sigma
        if self.sigma is None:
            self.sigma = np.ones_like(mu)

    @classmethod
    def from_action_space(cls, action_space):
        return cls(np.zeros(action_space.shape), np.ones(action_space.shape))

    def decay_sigma(self, rate=None):
        if rate is None:
            rate = self.decay
        if self.sigma > self.min_sigma:
            self.sigma *= rate
        else:
            self.sigma = self.min_sigma

    def __call__(self):
        noise = np.random.normal(self.mu, self.sigma, size=self.mu.shape)
        self.decay_sigma()
        return noise
