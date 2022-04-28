import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()

parser.add_argument("--filename", default='tq_flow_cos', type=str)
parser.add_argument("--model", default='gfn', type=str)
parser.add_argument("--reward_fn", default='cosine', type=str)
# True if full path; False if only path endpoints
parser.add_argument("--full_path", default=True, type=bool)
parser.add_argument("--gif", default=False, type=bool)
parser.add_argument("--num_iter", default=1000, type=int)

ndim = 2
horizon = 8


def main(args):
    with open(f'./results/{args.filename}.pkl', 'rb') as f:
        results = pickle.load(f)

    save_name = 'figures/'
    if not args.full_path and args.gif:
        save_name += 'endpoints_'
    if args.model == 'gfn':
        save_name += 'GFN_'
    elif args.model == 'mcmc':
        save_name += 'MCMC_'
    save_name += f'{args.filename.split()[0]}_{args.num_iter}'

    if args.gif:
        gif_and_png(results, save_name, args)
    else:
        png(results, save_name, args)


def gif_and_png(results, save_name, args):        
    if not args.full_path:
        modified = []
        for i in range(len(results) - 1):
            currX, currY = results[i]
            nextX, nextY = results[i+1]
            if (nextX == 0) and (nextY == 0):
                if (currX != 0) or (currY != 0):
                    modified.append((currX, currY))
        x, y = results[-1]
        modified.append((x, y))

        results = modified

    grid = np.zeros((9, 9))

    def animate(i):
        if i % 100 == 0:
            print(f'{i}/{len(results)} done')
        plt.clf()
        x, y = results[i]
        grid[x,y] += 1
        plt.imshow(grid)
        plt.title(f'{args.model.upper()} Grid Sampling with {args.reward_fn.capitalize()}-{ndim}-{horizon}')
        plt.colorbar()

    anim = FuncAnimation(plt.gcf(), animate, frames=args.num_iter, interval=100, blit=False)
    writergif = animation.PillowWriter(fps=120)
    #plt.show()

    anim.save(f'{save_name}.gif', writer=writergif)
    
    plt.savefig(f'{save_name}.png')

def png(results, save_name, args):
    grid = np.zeros((9, 9))

    for i in range(args.num_iter):
        x, y = results[i]
        grid[x,y] += 1

    plt.imshow(grid)
    plt.title(f'{args.model.upper()} Grid Sampling with {args.reward_fn.capitalize()}-{ndim}-{horizon}, n = {args.num_iter}')
    plt.colorbar()

    plt.savefig(f'{save_name}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)