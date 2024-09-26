import torch
from collections import deque

import numpy as np

class Transition:
    def __init__(self, state:torch.tensor, state_subset:torch.tensor, action:torch.tensor, reward:torch.tensor, next_state:torch.tensor, next_state_subset:torch.tensor):

        self.state = state
        self.state_subset = state_subset
        self.action = action
        self.next_state = next_state
        self.next_state_subset = next_state_subset
        self.reward = reward

class ExperienceReplay(object):
    def __init__(self, capacity):
        ''' 
        State: torch.Size(1, [1280, 200]) Actions: torch.Size([256, 1, 10, 200])
        Reward = (singular but expanded for 256)
        '''
        self.memory_buffer = deque([], maxlen=capacity)

    def add_memory(self, current_state:torch.tensor, actions:torch.tensor, next_state:torch.tensor, reward:torch.tensor):
        
        reward = torch.tensor([reward.item()] * actions.shape[0])
        for idx, sub_action in enumerate(actions):
            transition = Transition(
                state=current_state[0], #torch.Size([1280, 200])
                state_subset=current_state[1][idx], #torch.Size(pool_size, 200)
                action=sub_action.squeeze(0), #torch.Size([256, 1])
                next_state=next_state[0], #torch.Size([1280, 200])
                next_state_subset=next_state[1][idx], #torch.Size(pool_size, 200)
                reward=reward[idx] # torch.Size([256])
            )

            self.memory_buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        return [self.memory_buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.memory_buffer)