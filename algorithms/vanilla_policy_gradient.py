import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
import torch.distributions as dist
from collections import deque



class VanillaPolicyGradientModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim): 
        super(VanillaPolicyGradientModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = functional.softmax(self.fc2(x), dim=1)
        return x


class VanillaPolicyGradientAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, device, learning_rate) -> None:
        
        self.policy_net = VanillaPolicyGradientModel(state_representation_size, action_size, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.terminal_state_value = terminal_state_value
        self.device = device


    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        prob = self.policy_net(state_tensor)
        sampler = dist.Categorical(prob)
        action = sampler.sample()
        return action.item()
    
    def update_param(self, trajectory):
        """
        trajectory = {(s,a,s',r)}_{i=1}^n
        """
        reward_list = []
        state_list = []
        action_list = []

        for sasr_tuple in trajectory:
            state_list.append(sasr_tuple[0])
            action_list.append(sasr_tuple[1])
            reward_list.append(sasr_tuple[3])
        
        trajectory_length = len(trajectory)

        # calculate monte carlo return
        monte_carlo_return_list = []
        for i in range(trajectory_length):
            monte_carlo_return_list.append(np.sum(self.gamma**np.array(range(trajectory_length - i)) * np.array(reward_list[i:])) + 
                                           self.gamma**(trajectory_length - i) * self.terminal_state_value)
        return_tensor = torch.tensor(monte_carlo_return_list, dtype=torch.float32, device=self.device)

        # preprocess states and actions
        state_tensor = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.float32, device=self.device)

        probs_tensor = self.policy_net(state_tensor)
        sampler = dist.Categorical(probs_tensor)
        log_logits = sampler.log_prob(action_tensor)

        # turn gradient descent to gradient ascent by "-"
        loss = - torch.mean(log_logits * return_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class VanillaPolicyGradientMain(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, 
                 terminal_state_value, device, epoch, env, learning_rate, max_step_count) -> None:
        self.epoch = epoch
        self.env = env
        self.max_step_count = max_step_count
        self.agent = VanillaPolicyGradientAgent(state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, device, learning_rate)
    
    def training(self):
        accumulated_reward_list = []
        trajectory_length_list = []

        returns = deque(maxlen=100)
        count_que = deque(maxlen=100)
        for current_epoch in range(self.epoch):
            trajectory = []
            step_count = 0
            total_reward = 0
            # reset environment
            state = self.env.reset()
            while True:
                action = self.agent.choose_action(state)

                # use that action in the environment
                new_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # store state, action and reward
                if not done:
                    trajectory.append((state, action, new_state, reward))
                else:
                    trajectory.append((state, action, None, reward))

                state = new_state

                step_count += 1
                if step_count > self.max_step_count:
                    break
                
                if done:
                    break
            # if the agent doesn't directly go to terminal state, the skip training
            if len(trajectory) >= 2:
                # update the parameter of nn
                self.agent.update_param(trajectory)
            
            returns.append(total_reward)
            count_que.append(step_count)
            print("Episode: {:6d}  Avg. Return: {:6.2f}  Avg. Trainig Count {:6.2f}".format(current_epoch, np.mean(returns), np.mean(count_que)))

            accumulated_reward_list.append(total_reward)
            trajectory_length_list.append(step_count)
        
        return accumulated_reward_list, trajectory_length_list
    
    
    def testing(self):
        total_reward_list = []
        for _ in range(self.test_epoch):
            total_reward = 0
            # reset environment
            state = self.env.reset()
            while True:
                action = self.agent.choose_action(state)

                # use that action in the environment
                new_state, reward, done = self.env.step(action)
                total_reward += reward

                state = new_state
                if done:
                    break
            
            total_reward_list.append(total_reward)
        
        return np.mean(total_reward_list)


