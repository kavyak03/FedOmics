from copy import deepcopy
import torch

def federated_average(state_dicts):
    avg_state = deepcopy(state_dicts[0])
    for key in avg_state:
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state
