'''
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.distributions as dist
import numpy as np
from collections import deque



class ProximalPolicyOptimizationCriticModel(nn.Module):
    def __init__(self, state_representation_size, hidden_dim) -> None:
        super(ProximalPolicyOptimizationCriticModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        return x

class ProximalPolicyOptimizationPolicyModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim): 
        super(ProximalPolicyOptimizationPolicyModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = functional.softmax(self.fc2(x), dim=1)
        return x


class ProximalPolicyOptimizationAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, policy_learning_rate, inner_epoch, eps_clip) -> None:
        self.policy_net_fast = ProximalPolicyOptimizationPolicyModel(state_representation_size, action_size, hidden_dim).to(device)
        self.policy_net_slow = ProximalPolicyOptimizationPolicyModel(state_representation_size, action_size, hidden_dim)
        self.policy_net_slow.load_state_dict(self.policy_net_fast.state_dict())
        self.policy_net_slow.to(device)

        self.critic_net_fast = ProximalPolicyOptimizationCriticModel(state_representation_size, hidden_dim).to(device)
        self.critic_net_slow = ProximalPolicyOptimizationCriticModel(state_representation_size, hidden_dim)
        self.critic_net_slow.load_state_dict(self.critic_net_fast.state_dict())
        self.critic_net_slow.to(device)

        self.policy_net_optimizer = torch.optim.Adam(self.policy_net_fast.parameters(), lr=policy_learning_rate)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net_fast.parameters(), lr=critic_learning_rate)

        self.mse_loss = nn.MSELoss()
        
        self.gamma = gamma
        self.terminal_state_value = terminal_state_value
        self.device = device
        self.inner_epoch = inner_epoch
        self.eps_clip = eps_clip

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = self.policy_net_slow(state_tensor)
        action_dist = dist.Categorical(action_probs)
        action = action_dist.sample()

        return action.item()

    def update_param(self, trajectory):
        reward_list = []
        state_list = []
        action_list = []

        for sasr_tuple in trajectory:
            state_list.append(sasr_tuple[0])
            action_list.append(sasr_tuple[1])
            reward_list.append(sasr_tuple[3])

        # calculate monte carlo return
        monte_carlo_return_list = []
        trajectory_length = len(trajectory)
        for i in range(trajectory_length):
            monte_carlo_return_list.append(np.sum(self.gamma**np.array(range(trajectory_length - i)) * np.array(reward_list[i:])) + 
                                           self.gamma**(trajectory_length - i) * self.terminal_state_value)
        return_tensor_k = torch.tensor(monte_carlo_return_list, dtype=torch.float32, device=self.device)

        # preprocess states and actions
        state_tensor_k = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        action_tensor_k = torch.tensor(action_list, dtype=torch.int64, device=self.device)
        state_value_tensor_k = self.critic_net_slow(state_tensor_k)
        action_logprobs_tensor_k = dist.Categorical(self.policy_net_slow(state_tensor_k)).log_prob(action_tensor_k)

        # calculate advantages
        advantages_tensor_k = return_tensor_k - state_value_tensor_k

        # critic_loss_list = []
        # policy_loss_list = []

        # find argmax of the loss of policy network and argmin of the loss of critic network
        for _ in range(self.inner_epoch):

            # Evaluating old actions and values
            current_state_value = self.critic_net_fast(state_tensor_k).squeeze(-1)
            current_action_dist = dist.Categorical(self.policy_net_fast(state_tensor_k))
            current_action_logprobs = current_action_dist.log_prob(action_tensor_k)
            # current_action_logprobs = torch.log(self.policy_net_fast(state_tensor_k).gather(1, action_tensor_k.unsqueeze(-1)))
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(current_action_logprobs - action_logprobs_tensor_k.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages_tensor_k.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor_k.detach()

            # update actor parameter
            policy_net_loss = - torch.mean(torch.min(surr1, surr2))
            self.policy_net_optimizer.zero_grad()
            policy_net_loss.backward()
            self.policy_net_optimizer.step()

            # update critic parameter
            critic_net_loss = self.mse_loss(current_state_value, return_tensor_k.detach())
            self.critic_net_optimizer.zero_grad()
            critic_net_loss.backward()
            self.critic_net_optimizer.step()

            # p_loss = policy_net_loss.detach().to("cpu").numpy().tolist()
            # policy_loss_list.append(p_loss)
            # c_loss = critic_net_loss.detach().to("cpu").numpy().tolist()
            # critic_loss_list.append(c_loss)
        
        # print("policy loss list: " + str(policy_loss_list))
        # print("policy loss list max: " + str(max(policy_loss_list)) + " min: " + str(min(policy_loss_list)) + 
        #       " mean: " + str(np.mean(policy_loss_list)))
        # print("critic loss list max: " + str(max(critic_loss_list)) + " min: " + str(min(critic_loss_list)) + 
        #       " mean: " + str(np.mean(critic_loss_list)))

        # Copy new weights into old policy
        self.policy_net_slow.load_state_dict(self.policy_net_fast.state_dict())
        self.critic_net_slow.load_state_dict(self.critic_net_fast.state_dict())


class ProximalPolicyOptimizationMain(object):
    def __init__(self, env, epoch, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, device, 
                 critic_learning_rate, policy_learning_rate, inner_epoch, eps_clip, max_step_count) -> None:
        self.epoch = epoch
        self.env = env
        self.max_step_count = max_step_count
        self.agent = ProximalPolicyOptimizationAgent(state_representation_size, action_size, hidden_dim, gamma, terminal_state_value,
                                                      device, critic_learning_rate, policy_learning_rate, inner_epoch, eps_clip)

    def training(self):
        accumulated_reward_list = []
        trajectory_length_list = []

        returns = deque(maxlen=100)
        count_que = deque(maxlen=100)
        for current_epoch in range(self.epoch):
            trajectory = []
            total_reward = 0
            step_count = 0
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
                reward = +1
                total_reward += reward

                state = new_state
                if done:
                    break
            
            total_reward_list.append(total_reward)
        
        return np.mean(total_reward_list)


