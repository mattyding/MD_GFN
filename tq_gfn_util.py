'''
all code below is from tikquuss's Github repo
'''
import torch
from typing import List

'''
params
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_hid = 256
n_layers = 2
batch_size = 128
minus_inf = -1e8
uniform_PB = False

n_train_steps = 5000
H=8
ndim=2


def build_grid(ndim, H):
    # We have (H+1)^ndim points, each point being of dimension ndim.
    grid_shape = (H+1,)*ndim+(ndim,) # (H+1, ..., H+1, ndim)
    grid = torch.zeros(grid_shape)
    for i in range(ndim):
        grid_i = torch.linspace(start=0, end=H, steps=H+1)
        for _ in range(i): grid_i = grid_i.unsqueeze(1)
        grid[...,i] = grid_i
    #return grid.view((H+1)**ndim,-1) # ((H+1)*ndim, ndim)
    return grid

def make_mlp(l, act=torch.nn.LeakyReLU(), tail=[]):
    return torch.nn.Sequential(*(sum(
        [[torch.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

#re-execute the following cells each time H changes
def one_hot_embedding(states, num_classes=H+1):
    # states : (bs, ndim) -> (bs, embedding_dim), embedding_dim = num_classes x ndim
    assert num_classes >= H + 1
    return torch.nn.functional.one_hot(states, num_classes).view(states.shape[0],-1).float()


def get_modes_founds(first_visit : List[int], threshold : float = 2, max_steps : int = None) :  # float=R2
    """Calculates the modes (R >= threshold) found for each episode"""
    n = max_steps if max_steps is not None else n_train_steps
    xs = range(n)
    modes_founds = [0]*n
    for i in xs :
        states_i = list(set(np.where((0 <= first_visit) & (first_visit <= i))[0]))
        states_i = get_state(torch.LongTensor(states_i))
        r = reward_function(states_i)
        modes_founds[i] = (r >= threshold).sum().item()
    xs = [(i+1)*batch_size for i in xs]
    return modes_founds, xs

base_coordinates = (H+1)**torch.arange(ndim) # [(H+1)^0, ..., (H+1)^(ndim-1)]
def get_state_index(states):
    """
    This function allows to associate a unique index to each state of the grid environment.
    To find a state given its index, just invoke grid[state_index]
    > params : 
        * states ~ Tensor(num_states, ndim), batch_size can be zero (for single state)
    """
    state_index = (states*base_coordinates).sum().item()
    return state_index # (num_states,)

def get_state(indexes) :
    """
    This function allows to find a state given its index
    > params : 
        * indexes  ~ LongTensor(num_states,) or int (for single state)
    """
    return grid[indexes] # (num_states, ndim)