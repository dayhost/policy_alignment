import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.distributions as dist
from collections import deque


class AdvantageActorCriticCriticModel(nn.Module):
    def __init__(self, state_representation_size, hidden_dim) -> None:
        super(AdvantageActorCriticCriticModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        return x

class AdvantageActorCriticPolicyModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim): 
        super(AdvantageActorCriticPolicyModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = functional.softmax(self.fc2(x), dim=1)
        return x

class AdvantageActorCriticAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, policy_learning_rate) -> None:
        self.policy_net = AdvantageActorCriticPolicyModel(state_representation_size, action_size, hidden_dim).to(device)
        self.critic_net = AdvantageActorCriticCriticModel(state_representation_size, hidden_dim).to(device)
        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_learning_rate)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)
        
        self.mse_loss = nn.MSELoss()

        self.gamma = gamma
        self.device = device
        self.terminal_state_value = terminal_state_value
    
    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        action_dist = dist.Categorical(action_probs)
        action = action_dist.sample()

        return action.item()

    def update_param(self, trajectory):
        reward_list = []
        state_list = []
        action_list = []
        next_state_list = []
        non_final_next_state_list = []

        for sasr_tuple in trajectory:
            state_list.append(sasr_tuple[0])
            action_list.append(sasr_tuple[1])
            next_state_list.append(sasr_tuple[2])
            
            if sasr_tuple[2] is not None:
                non_final_next_state_list.append(sasr_tuple[2])
            
            reward_list.append(sasr_tuple[3])

        # preprocess states and actions
        state_tensor = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.int64, device=self.device)
        non_final_next_state_tensor = torch.tensor(non_final_next_state_list, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # 使用复制的model计算所有的V(s_t, a_t)，并且假设在s_t+1为最终的state的时候，V(s_t)=terminal_state_value
        # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state_list)), device=self.device, dtype=torch.bool
        )  
        next_state_value_tensor = torch.ones((len(trajectory), 1), device=self.device) * self.terminal_state_value
        # 获取state value，并将为Non-final的state value赋值
        next_state_value_tensor[non_final_mask] = self.critic_net(non_final_next_state_tensor).detach()

        # calculate TD error
        state_value_tensor = self.critic_net(state_tensor)
        td_error_tensor =   reward_tensor + self.gamma * next_state_value_tensor - state_value_tensor

        # update critic parameter
        critic_loss = torch.mean(td_error_tensor**2)
        self.critic_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_optimizer.step()

        # calculate action log probability
        # action_logprobs_tensor = torch.log(self.policy_net(state_tensor).gather(1, action_tensor.unsqueeze(-1)))
        action_logprobs_tensor = dist.Categorical(self.policy_net(state_tensor)).log_prob(action_tensor)
        policy_loss = -torch.mean(action_logprobs_tensor * td_error_tensor.detach())
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()


class AdvantageActorCriticMain(object):
    def __init__(self, env, epoch, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, policy_learning_rate, max_step_count) -> None:
        self.epoch = epoch
        self.env = env
        self.max_step_count = max_step_count
        self.agent = AdvantageActorCriticAgent(state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                                               device, critic_learning_rate, policy_learning_rate)

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


