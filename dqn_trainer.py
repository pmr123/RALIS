import torch
from torch.optim import AdamW, SGD

from model.dqn import QueryNetworkDQN
from transition import Transition, ExperienceReplay

class DQNTrainer:
    def __init__(self, target_net:QueryNetworkDQN, policy_net:QueryNetworkDQN):

        self.target_net = target_net
        self.policy_net = policy_net

        self._init_optimizer()

        self.batch_size = 80
        self.gamma = 0.9

    def _init_optimizer(self):

        print(f'Initailizing Optimizer for {self.policy_net.__class__.__name__}')
        param_dict = []

        param_dict.append({
            "params":self.policy_net.parameters(), "lr":5e-5, "model_name":"U-Net Encoder" 
        })

        self.optimizer = SGD(
            params=param_dict, weight_decay=0.001, nesterov=True, momentum=0.9
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )


    def train_double_dqn(self, memory:ExperienceReplay):
        self.policy_net.train()

        self.optimizer.zero_grad()
        transitions = memory.sample(batch_size=self.batch_size)    

        current_states = torch.stack([t.state for t in transitions]) #(bs, 1280, 200)
        current_states_subsets = torch.stack([t.state_subset for t in transitions]) #(bs, 10, 200)
        rewards = torch.stack([t.reward for t in transitions]) #(bs)
        actions = torch.stack([t.action for t in transitions]).unsqueeze(-1) #(bs, 1)  

        #(bs, 1, pool_size)
        q_values = self.policy_net(current_states.to(self.policy_net.device), current_states_subsets.to(self.policy_net.device))
        q_values = q_values.squeeze(1) #(bs, pool_size)
        
        current_q_values = q_values.gather(1, actions) #(bs, 1)
        current_q_values = current_q_values.squeeze(-1) #(bs)

        non_final_mask = torch.tensor([t.next_state is not None for t in transitions], device=self.policy_net.device, dtype=torch.bool)
        non_final_next_states = [t.next_state for t in transitions if t.next_state is not None]
        non_final_next_state_subsets = [t.next_state_subset for t in transitions if t.next_state_subset is not None]

        next_state_values = torch.zeros(len(transitions)).to(self.target_net.device)

        non_final_next_states = torch.stack(non_final_next_states)
        non_final_next_state_subsets = torch.stack(non_final_next_state_subsets)

        # Select best actions at next states from the policy net
        #torch.Size([12, 1, 10])
        next_state_action_values = self.policy_net(non_final_next_states.to(self.policy_net.device), non_final_next_state_subsets.to(self.policy_net.device)) 
        next_state_action_values = next_state_action_values.squeeze(1)
        #(bs)
        next_state_actions = next_state_action_values.max(1)[1]  # Indexes of the best actions

        # Evaluate these best actions with the target network
        next_state_q_values = self.target_net(non_final_next_states.to(self.target_net.device), non_final_next_state_subsets.to(self.target_net.device))
        next_state_q_values = next_state_q_values.squeeze(1)
        next_state_q_values = next_state_q_values.gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)

        next_state_values[non_final_mask] = next_state_q_values.detach()   
        
        expected_q_values = (next_state_values * self.gamma) + rewards.to(self.target_net.device)

        loss = torch.nn.SmoothL1Loss()(current_q_values, expected_q_values)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.lr_scheduler.step()

        return loss

    def train(self, memory:ExperienceReplay):
        self.policy_net.train()

        self.optimizer.zero_grad()

        transitions = memory.sample(batch_size=self.batch_size)        

        current_states = torch.stack([t.state for t in transitions]) #(bs, 1280, 200)
        current_states_subsets = torch.stack([t.state_subset for t in transitions]) #(bs, 10, 200)
        rewards = torch.stack([t.reward for t in transitions]) #(bs)
        actions = torch.stack([t.action for t in transitions]).unsqueeze(-1) #(bs, 1)

        #(bs, 1, pool_size)
        q_values = self.policy_net(current_states.to(self.policy_net.device), current_states_subsets.to(self.policy_net.device))
        q_values = q_values.squeeze(1) #(bs, pool_size)

        
        current_q_values = q_values.gather(1, actions) #(bs, 1)
        current_q_values = current_q_values.squeeze(-1) #(bs)

        non_final_mask = torch.tensor([t.next_state is not None for t in transitions], device=self.policy_net.device, dtype=torch.bool)
        next_states = torch.stack([t.next_state for t in transitions if t.next_state is not None])
        next_state_subsets = torch.stack([t.next_state_subset for t in transitions if t.next_state_subset is not None])

        next_state_q_values = torch.zeros(len(transitions)).to(self.target_net.device)
        target_q_values = self.target_net(
            next_states.to(self.target_net.device),
            next_state_subsets.to(self.target_net.device)
        ).squeeze(1) #(bs, pool_size)

        next_state_q_values[non_final_mask] = target_q_values.max(1)[0].detach() #(bs)

        expected_q_values = (next_state_q_values * self.gamma) + rewards.to(self.target_net.device)

        loss = torch.nn.SmoothL1Loss()(current_q_values, expected_q_values)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.lr_scheduler.step()

        return loss
