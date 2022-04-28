'''
modified code from tikquuss's Github repo
github citation 4/28/22 in README.md
'''
import argparse
import torch
import numpy as np
import tqdm
import pickle
from scipy.stats import norm
import matplotlib.pyplot as plt

from tq_gfn_util import *

''' hyperparameters '''
parser = argparse.ArgumentParser()
#parser.add_argument("--save_path", default='results/flow_insp_0.pkl.gz', type=str)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--reward_fn", default='cosine', type=str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_hid = 256
n_layers = 2
batch_size = 128
minus_inf = -1e8
uniform_PB = False
# reward function hyperparams
R0 = 1e-2
R1 = 2
R2 = 0.5


def func_cos_N(x, H):
    # note: adjust this function for different params
        ax = abs(x/H - 0.5)
        return R0 + ((np.cos(ax * 50) + 1) * norm.pdf(ax * 5)).prod(-1) * R1
def func_corners(x, H):
    ax = abs(x/H - 0.5)
    return R0 + (0.25 < ax).prod(-1) * R1  + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2


def main(args):
    H, ndim= args.horizon, args.ndim
    grid = build_grid(ndim, H)
    grid = grid.view((H+1)**ndim,-1)

    reward_function = func_cos_N if args.reward_fn == 'cosine' else func_corners

    all_rewards = reward_function(grid, H)
    true_dist = all_rewards.flatten().softmax(0).cpu().numpy()
    ar = all_rewards.reshape((H+1,)*ndim)
    plot_reward_fn(reward_function, ndim, H, f'{R0}-{R1}-{R2} {args.reward_fn.capitalize()} Reward Function, ndim={ndim}, H={H}')

    input_dim = ndim*(H+1) # embedding dim
    output_dim = 2*ndim+1 # ndim + 1 for P_F (+1 for stop action) and ndim for P_B 
    model_TB = make_mlp([input_dim] + [n_hid] * n_layers + [output_dim]).to(device)
    logZ_TB = torch.zeros((1,)).to(device) # log (initial state flow), Z = 1
    optimizer = torch.optim.Adam([ {'params':model_TB.parameters(), 'lr':0.001}, {'params':[logZ_TB], 'lr':0.1} ])
    logZ_TB.requires_grad_()
    losses_TB = []
    rewards_TB = []
    logZ_TB_list = []
    all_visited_TB = []
    first_visit_TB = -1 * np.ones_like(true_dist)
    l1log_TB = []
    for it in tqdm.trange(n_train_steps):
    
        # TB loss for each trajectory
        loss_TB = torch.zeros((batch_size,)).to(device)
        loss_TB += logZ_TB # see the equation above
        # finished trajectories
        dones = torch.full((batch_size,), False, dtype=torch.bool).to(device)
        # s_0
        states = torch.zeros((batch_size, ndim), dtype=torch.long).to(device)
        # actions chosen at each step : we can choose the coordinate to increment (0 ... ndim-1), or choose to return the current state as terminal (ndim)
        actions = None # (current_batch_size,)

        while torch.any(~dones):

            ### Forward pass ### 
            current_batch_size = (~dones).sum()
            non_terminal_states = states[~dones] # (current_batch_size, ndim)
            embed = one_hot_embedding(non_terminal_states) # (current_batch_size, input_dim)
            logits = model_TB(embed) # (current_batch_size, output_dim) 
            
            ### Backward Policy ### 
            PB_logits = logits[...,ndim+1:2*ndim+1] # (current_batch_size, ndim)
            PB_logits = PB_logits * (0 if uniform_PB else 1) # (current_batch_size, ndim)
            # Being in a edge cell -- (a zero coordinate), we can't move backward
            edge_mask = (non_terminal_states == 0).float() # (current_batch_size, ndim)
            logPB = (PB_logits + minus_inf*edge_mask).log_softmax(1) # (current_batch_size, ndim)
            # add -logPB to the loss
            if actions is not None: 
                """
                Gather along the parents' dimension (1) to select the logPB of the previously chosen actions, while avoiding the actions leading 
                to terminal states (action==ndim). The reason of using the previous chosen actions is that PB is calculated on the same trajectory as PF
                See below for the calculation of `action`. We avoid actions leading to terminal states because a terminal state can't be parent of another 
                state
                """
                loss_TB[~dones] -= logPB.gather(1, actions[actions != ndim].unsqueeze(1)).squeeze(1)

            ### Forward Policy ### 
            PF_logits = logits[...,:ndim+1] # (current_batch_size, ndim+1) 
            # Being in a edge cell ++ (a coordinate that is H), we can't move forward
            edge_mask = (non_terminal_states == H).float() # (current_batch_size, ndim)
            # but any cell can be a terminal cell
            stop_action_mask = torch.zeros((current_batch_size, 1), device=device) # (current_batch_size, 1)
            # Being in a edge cell, we can't move forward, but any cell can be a terminal cell
            PF_mask = torch.cat([edge_mask, stop_action_mask], 1) # (current_batch_size, ndim+1)
            # logPF (with mask)
            logPF = (PF_logits + minus_inf*PF_mask).log_softmax(1) # (current_batch_size, ndim+1)
            # choose next states
            sample_temperature = 1
            #exp_weight = 0.
            #sample_ins_probs = (1-exp_weight)*(logPF/sample_temperature).softmax(1) + exp_weight*(1-PF_mask) / (1-PF_mask+0.0000001).sum(1).unsqueeze(1) # (current_batch_size, ndim+1)
            sample_ins_probs = (logPF/sample_temperature).softmax(1) # (current_batch_size, ndim+1)
            #actions = torch.distributions.categorical.Categorical(probs = sample_ins_probs).sample() # (current_batch_size,)
            #actions = torch.multinomial(probs = sample_ins_probs, 1).squeeze(1) # (current_batch_size,) # (current_batch_size,)
            actions = sample_ins_probs.multinomial(1) # (current_batch_size,)
            # add logPF to the loss : gather along the children's dimension (1) to select the logPF for the chosen actions
            loss_TB[~dones] += logPF.gather(1, actions).squeeze(1)

            ### select terminal states ### 
            terminates = (actions==ndim).squeeze(1)
            for state in non_terminal_states[terminates]: 
                state_index = get_state_index(state.cpu())
                if first_visit_TB[state_index]<0: first_visit_TB[state_index] = it
                all_visited_TB.append(state_index)
            
            # Update dones
            dones[~dones] |= terminates

            # Update non completed trajectories : $s = (s^0, ..., s^i, ..., s^{n-1}) \rightarrow s' = (s^0, ..., s^i + 1, ..., s^{n-1})$
            with torch.no_grad():
                non_terminates = actions[~terminates]
                states[~dones] = states[~dones].scatter_add(1, non_terminates, torch.ones(non_terminates.shape, dtype=torch.long, device=device))
            
        R = reward_function(states.float(), H)
        loss_TB -= R.log()
        loss = (loss_TB**2).sum()/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_TB.append(loss.item())
        rewards_TB.append(R.mean().cpu())
        logZ_TB_list.append(logZ_TB.item())

        if it%100==0: 
            print('\nloss =', np.array(losses_TB[-100:]).mean(), 'logZ =', logZ_TB.item(), "R =", np.array(rewards_TB[-100:]).mean())
            emp_dist = np.bincount(all_visited_TB[-200000:], minlength=len(true_dist)).astype(float)
            emp_dist /= emp_dist.sum()
            l1 = np.abs(true_dist-emp_dist).mean()
            print('L1 =', l1)
            l1log_TB.append((len(all_visited_TB), l1))

    '''
    sampling
    '''
    #samples_TB = np.zeros((H+1,)*ndim)
    samples_TB = []
    for it in tqdm.trange(500):
        # finished trajectories
        dones = torch.full((batch_size,), False, dtype=torch.bool).to(device)
        # s_0
        states = torch.zeros((batch_size, ndim), dtype=torch.long).to(device)
        # actions chosen at each step : we can choose the coordinate to increment (0 ... ndim-1), or choose to return the current state as terminal (ndim)
        actions = None # (current_batch_size,)

        while torch.any(~dones):

            ### Forward pass ### 
            current_batch_size = (~dones).sum()
            non_terminal_states = states[~dones] # (current_batch_size, ndim)
            embed = one_hot_embedding(non_terminal_states) # (current_batch_size, input_dim)
            with torch.no_grad():
                logits = model_TB(embed) # (current_batch_size, output_dim) 
            
            ### Forward Policy ### 
            PF_logits = logits[...,:ndim+1] # (current_batch_size, ndim+1) 
            # Being in a edge cell ++ (a coordinate that is H), we can't move forward
            edge_mask = (non_terminal_states == H).float() # (current_batch_size, ndim)
            # but any cell can be a terminal cell
            stop_action_mask = torch.zeros((current_batch_size, 1), device=device) # (current_batch_size, 1)
            # Being in a edge cell, we can't move forward, but any cell can be a terminal cell
            PF_mask = torch.cat([edge_mask, stop_action_mask], 1) # (current_batch_size, ndim+1)
            # logPF (with mask)
            logPF = (PF_logits + minus_inf*PF_mask).log_softmax(1) # (current_batch_size, ndim+1)
            # choose next states
            sample_temperature = 1
            #exp_weight = 0.
            #sample_ins_probs = (1-exp_weight)*(logPF/sample_temperature).softmax(1) + exp_weight*(1-PF_mask) / (1-PF_mask+0.0000001).sum(1).unsqueeze(1) # (current_batch_size, ndim+1)
            sample_ins_probs = (logPF/sample_temperature).softmax(1) # (current_batch_size, ndim+1)
            #actions = torch.distributions.categorical.Categorical(probs = sample_ins_probs).sample() # (current_batch_size,)
            #actions = torch.multinomial(probs = sample_ins_probs, 1).squeeze(1) # (current_batch_size,) # (current_batch_size,)
            actions = sample_ins_probs.multinomial(1) # (current_batch_size,)
            
            ### select terminal states ### 
            terminates = (actions==ndim).squeeze(1)
            
            # Update dones
            dones[~dones] |= terminates
            
            # Update non completed trajectories : $s = (s^0, ..., s^i, ..., s^{n-1}) \rightarrow s' = (s^0, ..., s^i + 1, ..., s^{n-1})$
            with torch.no_grad():
                non_terminates = actions[~terminates]
                states[~dones] = states[~dones].scatter_add(1, non_terminates, torch.ones(non_terminates.shape, dtype=torch.long, device=device))

        for a,b in states: 
            samples_TB.append((a,b))
        
    pickle.dump(samples_TB, open(f'results/tq_gfn_{args.reward_fn}.pkl', 'wb'))
            



# debugging
def print_grid(grid, ndim):
    plt.imshow(grid) if ndim == 2 else plt.imshow(grid[0,0])
    plt.show()

def plot_reward_fn(reward_function, ndim, H, filename):
    grid = build_grid(ndim, H)
    grid = grid.view((H+1)**ndim,-1)
    all_rewards = reward_function(grid, H)
    ar = all_rewards.reshape((H+1,)*ndim)
    plt.imshow(ar)
    plt.title(filename)
    plt.savefig(f'figures/{R0}-{R1}-{R2}_{filename.split()[1].lower()}_{ndim}_{H}.png')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)