import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()

parser.add_argument("--filename", default='toy_gfn_cosN', type=str)
# True if full path; False if only path endpoints
parser.add_argument("--full_path", default=True, type=bool)


def main(args):
    with open(f'./results/{args.filename}.pkl', 'rb') as f:
        results = pickle.load(f)
    #losses = results['losses']
    #params = results['params']
    all_visited = results['visited']
    #emp_dist_loss = results['emp_dist_loss']
    #true_d = results['true_d']
    #args = results['args']

    if not args.full_path:
        modified_visited = []
        for i in range(len(all_visited) - 1):
            currX, currY = all_visited[i]
            nextX, nextY = all_visited[i+1]
            if (nextX == 0) and (nextY == 0):
                if (currX != 0) or (currY != 0):
                    modified_visited.append((currX, currY))
        x, y = all_visited[-1]
        modified_visited.append((x, y))

        all_visited = modified_visited

    grid = np.zeros((8,8))

    def animate(i):
        if i % 100 == 0:
            print(f'{i}/{len(all_visited)} done')
        plt.clf()
        x, y = all_visited[i]
        grid[x,y] += 1
        plt.imshow(grid)
        plt.title('GFN Sampling on Grid with Cosine Reward Function')
        plt.colorbar()

    anim = FuncAnimation(plt.gcf(), animate, frames=1000, interval=100, blit=False)
    writergif = animation.PillowWriter(fps=120)
    #plt.show()
    save_name = f"figures/{args.filename}" if args.full_path else f"figures/endpoints_{args.filename}"

    anim.save(f'figures/{save_name}.gif', writer=writergif)
    
    plt.savefig(f'figures/{save_name}.png')



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)