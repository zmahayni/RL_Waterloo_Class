import collections
import numpy as np
import random
import torch


class ReplayBuffer:
    def __init__(self, N, recurrent=False):
        self.buf = collections.deque(maxlen=N)
        self.recurrent = recurrent

    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []

        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]
            A += [a]
            R += [r]
            S2 += [s2]
            D += [d]

        if type(A[0]) == int:
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        elif type(A[0]) == float:
            return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
        else:
            return t.f(S), torch.stack(A), t.f(R), t.f(S2), t.i(D)
