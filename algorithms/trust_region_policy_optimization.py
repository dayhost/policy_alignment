import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.distributions as dist
from collections import deque


class TrustRegionPolicyOptimziationCriticModel(nn.Module):
    def __init__(self, state_representation_size, hidden_dim) -> None:
        super(TrustRegionPolicyOptimziationCriticModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        return x


class TrustRegionPolicyOptimziationPolicyModel(nn.Module):
    def __init__(self, state_representation_size, action_size, hidden_dim): 
        super(TrustRegionPolicyOptimziationPolicyModel, self).__init__()
        self.fc1 = nn.Linear(state_representation_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state_tensor):
        x = functional.relu(self.fc1(state_tensor))
        x = functional.softmax(self.fc2(x), dim=1)
        return x
    

class TrustRegionPolicyOptimziationAgent(object):
    def __init__(self, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, max_kl_distance, alpha) -> None:
        
        self.critic_net = TrustRegionPolicyOptimziationCriticModel(state_representation_size, hidden_dim).to(device)
        self.policy_net = TrustRegionPolicyOptimziationPolicyModel(state_representation_size, action_size, hidden_dim).to(device)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)
        
        self.alpha = alpha

        self.gamma = gamma
        self.terminal_state_value = terminal_state_value
        self.device = device

        self.mse_loss = nn.MSELoss()
        self.max_kl_distance = max_kl_distance

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        action_dist = dist.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()
    
    def _calculate_surrogate_objective(self, state_tensor, action_tensor, advantage_tensor, old_log_probs, policy_net):
        # 计算策略网络损失函数对应的loss
        log_probs = torch.log(policy_net(state_tensor).gather(1, action_tensor.unsqueeze(-1)))
        # log_probs = torch.distributions.Categorical(policy_net(state_tensor)).log_prob(action_tensor)
        ratio = torch.exp(log_probs - old_log_probs.detach())
        
        return torch.mean(ratio * advantage_tensor.detach())

    def _hessian_matrix_vector_product(self, state_tensor, vector):
        # 计算黑塞矩阵和一个向量的乘积
        action_dist_tensor = self.policy_net(state_tensor)
        new_action_dists = torch.distributions.Categorical(action_dist_tensor)
        old_action_dists = torch.distributions.Categorical(action_dist_tensor.detach())
        
        # 计算平均KL距离
        kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)) 
        # 这里需要create graph，设置这个为True主要是为了计算KL Div的Hessian，在设置了create graph=true的情况下就可以再基于gradient计算hessian
        kl_grad = torch.autograd.grad(kl_div, self.policy_net.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        hessian_vector_prod = torch.autograd.grad(kl_grad_vector_product, self.policy_net.parameters())
        flatten_hessian_vector_prod = torch.cat([grad.view(-1) for grad in hessian_vector_prod])
        
        return flatten_hessian_vector_prod
    
    def _conjugate_gradient_method_find_optimal_direction(self, state_tensor, gradient_0, early_stop_critiera=1e-10, max_iteration=10):
        
        x = torch.zeros_like(gradient_0)
        p = gradient_0.clone()
        r = gradient_0.clone()
        rdotr = torch.dot(r, r)

        for _ in range(max_iteration):
            Hp = self._hessian_matrix_vector_product(state_tensor, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)

            if new_rdotr < early_stop_critiera:
                break
            
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    

    def _backtracking_line_search(self, state_tensor, action_tensor, advantage_tensor, update_direction, 
                                  old_action_logprobs_tensor, old_action_dists):
            
            old_param = torch.nn.utils.convert_parameters.parameters_to_vector(self.policy_net.parameters())
            old_objective = self._calculate_surrogate_objective(state_tensor, action_tensor, advantage_tensor, 
                                                                old_action_logprobs_tensor, self.policy_net)
            
            try:
                for i in range(15):  # 线性搜索主循环
                    coef = self.alpha ** i
                    new_param = old_param + coef * update_direction
                    new_policy_net = copy.deepcopy(self.policy_net)
                    
                    # 将更新后的参数放入新的policy_net中
                    torch.nn.utils.convert_parameters.vector_to_parameters(new_param, new_policy_net.parameters())
                    
                    # 计算新的loss和KL divergence
                    new_action_probs = new_policy_net(state_tensor)
                    new_action_dists = torch.distributions.Categorical(new_action_probs)

                    kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
                    new_objective = self._calculate_surrogate_objective(state_tensor, action_tensor, advantage_tensor, 
                                                                        old_action_logprobs_tensor, new_policy_net)
                    
                    if new_objective > old_objective and kl_div < self.max_kl_distance:
                        return new_param
            except:
                print("nan exist!!")
            
            return old_param

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
        state_tensor = torch.tensor(np.array(state_list), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.int64, device=self.device)

        # calculate advantages using monte carlo return
        monte_carlo_return_list = []
        trajectory_length = len(trajectory)
        for i in range(trajectory_length):
            monte_carlo_return_list.append(np.sum(self.gamma**np.array(range(trajectory_length - i)) * np.array(reward_list[i:])) + 
                                           self.gamma**(trajectory_length - i) * self.terminal_state_value)
        return_tensor = torch.tensor(monte_carlo_return_list, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        state_value_tensor = self.critic_net(state_tensor)
        advantage_tensor = return_tensor - state_value_tensor

        # non_final_next_state_tensor = torch.tensor(non_final_next_state_list, device=self.device)
        # reward_tensor = torch.tensor(reward_list, device=self.device).unsqueeze(-1)
        # # 使用复制的model计算所有的V(s_t, a_t)，并且假设在s_t+1为最终的state的时候，V(s_t)=terminal_state_value
        # # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
        # non_final_mask = torch.tensor(
        #     tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
        # )  
        # next_state_value_tensor = torch.ones((len(trajectory), 1), device=device) * self.terminal_state_value
        # # 获取state value，并将为Non-final的state value赋值
        # next_state_value_tensor[non_final_mask] = self.critic(non_final_next_state_tensor)

        # # calculate TD error
        # td_target = reward_tensor + self.gamma * next_state_value_tensor
        # td_error_tensor =  td_target - self.critic(state_tensor)
        # advantage = compute_advantage(self.gamma, self.lmbda, td_error_tensor.cpu()).to(self.device)  # 使用GAM的方式计算advantage

        critic_loss = self.mse_loss(state_value_tensor, return_tensor.detach())
        self.critic_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_optimizer.step()  # 更新价值函数
                  
        # 更新策略函数
        old_action_dists = torch.distributions.Categorical(self.policy_net(state_tensor).detach())
        # 用torch.log计算的和用distributions计算的log prob对于backwardprop有不同的影响，虽然两个函数求出来的结果是一样的，但是收敛的速度不一样！！
        # old_log_probs = torch.log(self.policy_net(state_tensor).gather(1, action_tensor.unsqueeze(-1))).detach()
        old_log_probs = torch.distributions.Categorical(self.policy_net(state_tensor)).log_prob(action_tensor).detach()
        
        # 用共轭梯度法计算x = H^(-1)g
        surrogate_obj = self._calculate_surrogate_objective(state_tensor, action_tensor, advantage_tensor, old_log_probs, self.policy_net)
        grads = torch.autograd.grad(surrogate_obj, self.policy_net.parameters())
        flatten_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self._conjugate_gradient_method_find_optimal_direction(state_tensor, flatten_grad)

        Hd = self._hessian_matrix_vector_product(state_tensor, flatten_grad)
        max_coef = torch.sqrt(2 * self.max_kl_distance / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self._backtracking_line_search(state_tensor, action_tensor, advantage_tensor, 
                                                  descent_direction * max_coef, old_log_probs, old_action_dists)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.policy_net.parameters())  # 用线性搜索后的参数更新策略 
      

class TrustRegionPolicyOptimziationMain(object):
    def __init__(self, env, epoch, state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, max_kl_distance, alpha, max_step_count) -> None:
        self.epoch = epoch
        self.env = env
        self.max_step_count = max_step_count
        self.agent = TrustRegionPolicyOptimziationAgent(state_representation_size, action_size, hidden_dim, gamma, terminal_state_value, 
                 device, critic_learning_rate, max_kl_distance, alpha)

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
                total_reward += reward

                state = new_state
                if done:
                    break
            
            total_reward_list.append(total_reward)
        
        return np.mean(total_reward_list)


