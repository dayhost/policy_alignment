import torch
import random
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as functional
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_tuple = []
        self.buffer_size = buffer_size
        self.data_num_count = 0

    def insert_data_tuple(self, current_state, current_action, next_state, reward):
        """
        本函数采用循环链表插入的方法，即当Buffer满了之后，会从index0处开始进行数据覆盖
        :param current_state:
        :param current_action:
        :param next_state:
        :param reward:
        :return:
        """
        if self.get_buffer_length() < self.buffer_size:
            self.buffer_tuple.append((current_state, current_action, next_state, reward))
        else:
            self.buffer_tuple[self.data_num_count % self.buffer_size] = \
                (current_state, current_action, next_state, reward)
        self.data_num_count += 1

    def get_shuffle_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if batch_size < len(self.buffer_tuple):
            result = random.sample(self.buffer_tuple, batch_size)
        else:
            result = random.sample(self.buffer_tuple, len(self.buffer_tuple))

        return result

    def get_buffer_length(self):
        return len(self.buffer_tuple)


class DeepQNetworkModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim) -> None:
        super(DeepQNetworkModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)
    
    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        return x


class DeepQNetworkAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, 
                 terminal_state_value, device, learning_rate, epsilon, target_change_count, exploration_decay_count) -> None:
        
        self.model = DeepQNetworkModel(state_representation_size, action_size, hidden_dim).to(device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.action_size = action_size

        self.mse_loss = nn.MSELoss()
        self.gamma = gamma
        self.terminal_state_value = terminal_state_value
        self.device = device

        self.count = 0
        self.epsilon = epsilon
        self.target_change_count = target_change_count
        self.exploration_decay_count = exploration_decay_count

    def choose_action(self, state):
        # 每更新200次，探索概率变成原先的一半
        if self.count % 200 == 0 and self.count != 0:
            self.epsilon = self.epsilon * (1/2) ** int(self.count / self.exploration_decay_count)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.model(state).argmax().item()
        return action

    def update_param(self, sample_data_set):
        self.count += 1

        reward_list = []
        state_list = []
        action_list = []
        next_state_list = []
        non_final_next_state_list = []

        for sasr_tuple in sample_data_set:
            state_list.append(sasr_tuple[0])
            action_list.append(sasr_tuple[1])
            next_state_list.append(sasr_tuple[2])
            
            if sasr_tuple[2] is not None:
                non_final_next_state_list.append(sasr_tuple[2])
            
            reward_list.append(sasr_tuple[3])

        # preprocess states and actions
        state_tensor = torch.tensor(np.array(state_list), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.int64, device=self.device)
        non_final_next_state_tensor = torch.tensor(non_final_next_state_list, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device)

        # 使用复制的model计算所有的Q(s_t+1, a_t+1)，并且假设在s_t+1为最终的state的时候，Q(s_t+1, a_t+1)=0
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state_list)), device=self.device, dtype=torch.bool
        )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
        next_state_action_value = torch.ones(len(sample_data_set), device=self.device) * self.terminal_state_value
        # 获取maximum state action value，并将为Non-final的state value赋值
        next_state_action_value[non_final_mask] = self.target_model(non_final_next_state_tensor).max(1)[0] 
        
        # 使用原始的model计算所有的Q(s_t, a_t)，并且按照选择的action，将对应的action value选出来
        current_state_action_value_raw = self.model(state_tensor)
        current_state_action_value = current_state_action_value_raw.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)

        # 计算损失函数
        target_value = reward_tensor + self.gamma * next_state_action_value
        loss = self.mse_loss(current_state_action_value, target_value.detach())  # 以current state value作为学习对象

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()

        if self.count % self.target_change_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # 返回当前函数的loss
        return loss.detach().to("cpu").numpy().tolist()



class DeepQNetworkMain(object):
    def __init__(self, env, epoch, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, device, 
                 learning_rate, buffer_size, batch_size, epsilon, target_change_count, max_step_count, exploration_decay_count) -> None:
        self.epoch = epoch
        self.env = env
        self.batch_size = batch_size
        self.max_step_count = max_step_count
        self.memory = ReplayBuffer(buffer_size)
        self.agent = DeepQNetworkAgent(state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, device, 
                                       learning_rate, epsilon, target_change_count, exploration_decay_count)

    def training(self):
        accumulated_reward_list = []
        trajectory_length_list = []

        returns = deque(maxlen=100)
        count_que = deque(maxlen=100)
        for current_epoch in range(self.epoch):
            total_reward = 0
            # reset environment
            step_count = 0
            state = self.env.reset()
            while True:
                action = self.agent.choose_action(state)

                # use that action in the environment
                new_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # store state, action and reward
                if not done:
                    self.memory.insert_data_tuple(current_state=state, current_action=action, next_state=new_state, reward=reward)
                else:
                    self.memory.insert_data_tuple(current_state=state, current_action=action, next_state=None, reward=reward)

                if self.memory.get_buffer_length() > self.batch_size:
                    sample_data_set = self.memory.get_shuffle_batch_data(batch_size=self.batch_size)
                    self.agent.update_param(sample_data_set)

                step_count += 1
                if step_count > self.max_step_count:
                    break

                state = new_state
                if done:
                    break
    
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


