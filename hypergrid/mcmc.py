'''
File: hypergrid-matt/mcmc.py
----------------------------------
This file contains an implementation of a MCMC sampler for the hypergrid model.
'''
from hypergrid import HyperGrid

import numpy as np


class MH_Sampler():
    def __init__(self, args, envs):
        self.batch = [i.reset() for i in envs] # The N MCMC chains
        self.bufsize = args.bufsize
        self.nactions = args.ndim*2  # ergodic
        self.model = None

    def parameters(self):
        return []

    def sample_many(self, mbsize, all_visited):
        r = np.float32([i[1] for i in self.batch])
        a = np.random.randint(0, self.nactions, self.bufsize)
        steps = [self.envs[j].step(a[j], s=self.batch[j][2]) for j in range(self.bufsize)]
        rp = np.float32([i[1] for i in steps])
        A = rp / r
        U = np.random.uniform(0,1,self.bufsize)
        for j in range(self.bufsize):
            if A[j] > U[j]: # Accept
                self.batch[j] = (None, rp[j], steps[j][2])
                all_visited.append(tuple(steps[j][2]))
        return []

    def learn_from(self, *a):
        return None


def main():
    HyperGrid(2, 10).showGrid()

if __name__ == '__main__':
    main()