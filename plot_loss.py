import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tq_gfn_grid import func_cos_N
from tq_gfn_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--reward_fn', default='cosine', type=str)

def main(args):
    with open(f'results/tq_gfn_{args.reward_fn}.pkl', 'rb') as f:
        gfn_data = pickle.load(f)
        f.close()
    """
    with open('results/tq_mcmc_{args.reward_fn}.pkl', 'rb') as f:
        mcmc_data = pickle.load(f)
        f.close()
    """
    grid = build_grid(ndim, H)
    grid = grid.view((H+1)**ndim,-1)
    true_distrb = np.reshape(func_cos_N(grid, H).numpy(), (9, 9))
    true_distrb /= sum(true_distrb)

    losses = []

    gfn_grid = np.zeros((9, 9))
    for i, (x, y) in enumerate(gfn_data):
        gfn_grid[x, y] += 1
        #if i % 100 == 0:
        temp = gfn_grid.copy()
        temp /= sum(temp)
        loss = np.sum((temp - true_distrb) ** 2)
        #print(f'{i} loss: {loss}')
        losses.append(loss)
            
    gfn_grid = gfn_grid / np.sum(gfn_grid)  # normalize

    #diff = gfn_grid - true_distrb
    #plt.imshow(diff, cmap='RdBu', interpolation='nearest')
    #plt.colorbar()
    #plt.title(f"Difference between True and GFN distributions on {args.reward_fn.capitalize()} Function")
    #plt.savefig(f"figures/gfn_diff_{args.reward_fn}.png")
    print("Total Loss: {}".format(np.sum((gfn_grid - true_distrb) ** 2)))
    plt.plot(losses)
    plt.title(f"Norm Squared Sampling Losses on {args.reward_fn.capitalize()} Function")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"figures/gfn_iterloss_{args.reward_fn}.png")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)