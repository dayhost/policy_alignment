import torch
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.distributions as dist
from env.mountain_car import MountainCarCustomEnv
from env.acrobot import AcrobotCustomEnv
from algorithms.deep_q_network import DeepQNetworkMain


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CriticModel(nn.Module):
    def __init__(self, state_representation_size, hidden_dim) -> None:
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        return x

class PolicyModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim): 
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = functional.softmax(self.fc2(x), dim=1)
        return x

class ImitationLearningAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim,
                 device, critic_learning_rate, policy_learning_rate) -> None:
        self.policy_net = PolicyModel(state_representation_size, action_size, hidden_dim).to(device)
        self.critic_net = CriticModel(state_representation_size, hidden_dim).to(device)
        
        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_learning_rate)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)    
        
        self.mse_loss = nn.MSELoss()
        self.device = device

    def update_param(self, sample_data, imitate_model):
        state_list = []
        action_list = []

        for sasr_tuple in sample_data:
            state_list.append(sasr_tuple[0])
            action_list.append(sasr_tuple[1])

        # preprocess states and actions
        state_tensor = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.int64, device=self.device)

        # update critic parameter
        imitate_state_value = imitate_model(state_tensor).max(1)[0]
        current_state_value = self.critic_net(state_tensor).squeeze(-1)
        # update the state value by MSE loss
        critic_loss = self.mse_loss(current_state_value, imitate_state_value)
        self.critic_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_optimizer.step()

        # calculate action log probability
        action_logprobs_tensor = dist.Categorical(self.policy_net(state_tensor)).log_prob(action_tensor)
        # maximize policy log likelihood
        policy_loss = -torch.mean(action_logprobs_tensor)
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()

        print("policy loss: " + str(policy_loss.detach().data) + ", critic loss:" + str(critic_loss.detach().data))

    def save_model(self, filename, path):
        torch.save({"policy_net": self.policy_net.state_dict(), "critic_net": self.critic_net.state_dict()}, 
                   path + "/" + filename + ".pl")



def imitation_learning(random_seed, env_name):

    terminal_state_value = 0
    gamma = 0.99
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    critic_learning_rate = 1e-2
    policy_learning_rate = 1e-3
    hidden_dim = 64
    max_step_length = 500
    epoch_num = 100
    origin_training_epoch = 2000
    imitation_training_epoch = 10000

    if env_name == "mountain_car":
        
        env = MountainCarCustomEnv(seed=random_seed)
        state_representation_size = env.state_representation_size
        action_size = env.action_size

    elif env_name == "acrobot":
        
        env = AcrobotCustomEnv(seed=random_seed)
        state_representation_size = env.state_representation_size
        action_size = env.action_size
        

    setup_seed(random_seed)

    buffer_size = 10000
    batch_size = 128
    explore_epsilon = 0.2
    target_change_num = 300
    exploration_decay_count = 300
    alg_main = DeepQNetworkMain(env=env, epoch=origin_training_epoch, state_representation_size=state_representation_size, 
                                action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                terminal_state_value=terminal_state_value, device=device, 
                                learning_rate=critic_learning_rate, buffer_size=buffer_size, batch_size=batch_size, 
                                epsilon=explore_epsilon, target_change_count=target_change_num, max_step_count=max_step_length,
                                exploration_decay_count=exploration_decay_count)
    alg_main.training()

    sample_data = []
    for _ in range(epoch_num):
        state = env.reset()
        step_count = 0
        while True:
            action = alg_main.agent.choose_action(state)

            # use that action in the environment
            new_state, _, done = env.step(action)
            
            # store state, action and reward
            sample_data.append((state, action))

            step_count += 1
            if step_count > max_step_length:
                break

            state = new_state
            if done:
                break

    imitation_learning_agent = ImitationLearningAgent(
        state_representation_size=state_representation_size, action_size=action_size, hidden_dim=hidden_dim, 
        device=device, critic_learning_rate=critic_learning_rate, policy_learning_rate=policy_learning_rate)
    
    for _ in range(imitation_training_epoch):
        batch_sample_data = []
        choice_idx_list = np.random.choice(len(sample_data), size=64, replace=False)
        for i in choice_idx_list:
            batch_sample_data.append(sample_data[i])
        imitation_learning_agent.update_param(sample_data=batch_sample_data, imitate_model=alg_main.agent.model)
    
    path = "./model"
    file_name = env_name + "_model"
    imitation_learning_agent.save_model(path=path, filename=file_name)


def main():    
    random_seed = 0
    for env_name in ["acrobot", "mountain_car"]:
        imitation_learning(random_seed, env_name)
                
                        
if __name__ == "__main__":
    main()
    
