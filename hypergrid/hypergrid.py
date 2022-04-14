'''
File: hypergrid-matt/hypergrid.py
----------------------------------
This file contains my implementation of a hypergrid environment.
'''
import itertools

import numpy as np
import matplotlib.pyplot as plt

# reward function hyperparams (r0 << r1 < r2)
r0 = 0.01
r1 = 0.5
r2 = 2

class HyperGrid():
    def __init__(self, n: int, H: int):
        '''
        params: n: int, dimension of the grid
                H: int, length of each side of the grid
        '''
        self.n = n
        self.H = H
        self.grid = np.empty((self.H+1,) * n)

        self.Z = 0  # for validation

        for gridIndex in itertools.product(range(H+1), repeat=n):
            self.grid[gridIndex] = reward_fn(np.array(gridIndex), self.H)
            self.Z += self.grid[gridIndex]

        logZ = np.log(self.Z)
        self.probs = np.log(self.grid) - logZ
        print(self.probs)
        
    def showGrid(self):
        if self.n == 2 :
            plt.imshow(self.grid)
        else :
            plt.imshow(self.grid[0,0])
        plt.colorbar()
        plt.show()


def reward_fn(state: np.ndarray, H: int) -> float:
        '''
        corner reward functions from EBengio paper
        https://arxiv.org/pdf/2106.04399.pdf, page 7
        '''
        dist = abs((state / H) - 0.5)
        return r0 + (r1 * np.all(0.25 <= dist)) + (r2 * np.all(np.logical_and(0.3 <= dist, 0.4 >= dist)))