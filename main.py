import torch
import pickle
import numpy as np
import random

from env.cartpole import CartPoleCustomEnv
from env.mountain_car import MountainCarCustomEnv
from env.acrobot import AcrobotCustomEnv
from env.stochastic_MDP_negative_reward import StochasticNegativeMDP
from env.stochastic_MDP_positive_reward import StochasticPositiveMDP

from algorithms.vanilla_policy_gradient import VanillaPolicyGradientMain
from algorithms.deep_q_network import DeepQNetworkMain
from algorithms.proximal_policy_optimization import ProximalPolicyOptimizationMain
from algorithms.advantage_actor_critic import AdvantageActorCriticMain
from algorithms.trust_region_policy_optimization import TrustRegionPolicyOptimziationMain

from visualization.plot_returns import plt_accumulate_reward_dynamics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_reward_list(random_seed, env_name, algorithm_name, training_epoch, zero_non_zero_flag, load_model=False):

    gamma = 0.99
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    critic_learning_rate = 1e-2
    policy_learning_rate = 1e-3
    hidden_dim = 64
    max_step_length = 500

    if env_name == "cart_pole":
        
        env = CartPoleCustomEnv(seed=random_seed)
        state_representation_size = env.state_representation_size
        action_size = env.action_size

        if zero_non_zero_flag == "non_zero":
            terminal_state_value = 1/(1-gamma) * (10/9)
        elif zero_non_zero_flag == "zero":
            terminal_state_value = 0
    
    elif env_name == "mountain_car":
        
        env = MountainCarCustomEnv(seed=random_seed)
        state_representation_size = env.state_representation_size
        action_size = env.action_size

        if zero_non_zero_flag == "non_zero":
            terminal_state_value = -1/(1-gamma) * (10/9)
        elif zero_non_zero_flag == "zero":
            terminal_state_value = 0

    elif env_name == "acrobot":
        
        env = AcrobotCustomEnv(seed=random_seed)
        state_representation_size = env.state_representation_size
        action_size = env.action_size

        if zero_non_zero_flag == "non_zero":
            terminal_state_value = -1/(1-gamma) * (10/9)
        elif zero_non_zero_flag == "zero":
            terminal_state_value = 0
    
    elif env_name == "stochatsic_negative_mdp":
        
        r_min = -20
        r_max = -0.1
        state_size = state_representation_size = 10
        terminal_state_size = 1 
        action_size = 3
        env = StochasticNegativeMDP(state_size=state_representation_size, terminal_state_size=terminal_state_size, 
                                    action_size=action_size, random_seed=random_seed, r_min=r_min, r_max=r_max)
        
        if zero_non_zero_flag == "non_zero":
            terminal_state_value = (10/9) * (r_max - r_min) / \
                (gamma ** (state_size - 1) * (1 - gamma) * env.get_minimal_positive_transition_probability())
        elif zero_non_zero_flag == "zero":
            terminal_state_value = 0

    
    elif env_name == "stochatsic_positive_mdp":
        
        r_min = 0.1
        r_max = 20
        state_size = state_representation_size = 10
        terminal_state_size = 1
        action_size = 3
        env = StochasticPositiveMDP(state_size=state_representation_size, terminal_state_size=terminal_state_size, 
                                    action_size=action_size, random_seed=random_seed, r_min=r_min, r_max=r_max)
        
        if zero_non_zero_flag == "zero":
            terminal_state_value = 0
        elif zero_non_zero_flag == "non_zero":
            terminal_state_value = (10/9) * (r_min - r_max) / \
                (gamma ** (state_size - 1) * (1 - gamma) * env.get_minimal_positive_transition_probability())
        

    setup_seed(random_seed)

    if algorithm_name == "dqn":
        buffer_size = 50000
        batch_size = 128
        if env_name == "stochatsic_positive_mdp":
            explore_epsilon = 0.5
        else:
            explore_epsilon = 0.2
        target_change_num = 300
        exploration_decay_count = 300
        alg_main = DeepQNetworkMain(env=env, epoch=training_epoch, state_representation_size=state_representation_size, 
                                    action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                    terminal_state_value=terminal_state_value, device=device, 
                                    learning_rate=critic_learning_rate, buffer_size=buffer_size, batch_size=batch_size, 
                                    epsilon=explore_epsilon, target_change_count=target_change_num, max_step_count=max_step_length,
                                    exploration_decay_count=exploration_decay_count)
        
    elif algorithm_name == "vpg":
        alg_main = VanillaPolicyGradientMain(state_representation_size=state_representation_size, 
                                             action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                             terminal_state_value=terminal_state_value, device=device, 
                                             epoch=training_epoch, env=env, learning_rate=policy_learning_rate,
                                             max_step_count=max_step_length)
        
        if load_model is True:
            pickle_data = torch.load("./model/" + env_name + "_model.pl")
            alg_main.agent.policy_net.load_state_dict(pickle_data["policy_net"])
            alg_main.agent.policy_net.to(device)
            alg_main.agent.optimizer = torch.optim.Adam(alg_main.agent.policy_net.parameters(), lr=policy_learning_rate)
    
    elif algorithm_name == "a2c":
        alg_main = AdvantageActorCriticMain(env=env, epoch=training_epoch, state_representation_size=state_representation_size, 
                                            action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                            terminal_state_value=terminal_state_value, device=device, 
                                            critic_learning_rate=critic_learning_rate, policy_learning_rate=policy_learning_rate,
                                            max_step_count=max_step_length)
        
        if load_model is True:
            pickle_data = torch.load("./model/" + env_name + "_model.pl")
            alg_main.agent.policy_net.load_state_dict(pickle_data["policy_net"])
            alg_main.agent.policy_net.to(device)
            alg_main.agent.critic_net.load_state_dict(pickle_data["critic_net"])
            alg_main.agent.critic_net.to(device)

            alg_main.agent.policy_net_optimizer = torch.optim.Adam(alg_main.agent.policy_net.parameters(), lr=policy_learning_rate)
            alg_main.agent.critic_net_optimizer = torch.optim.Adam(alg_main.agent.critic_net.parameters(), lr=critic_learning_rate)
        
    elif algorithm_name == "ppo":
        inner_epoch = 20
        if load_model is True:
            epsilon_clip = 0.05
        else:
            epsilon_clip = 0.2
        alg_main = ProximalPolicyOptimizationMain(env=env, epoch=training_epoch, state_representation_size=state_representation_size, 
                                                  action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                                  terminal_state_value=terminal_state_value, device=device, 
                                                  critic_learning_rate=critic_learning_rate, 
                                                  policy_learning_rate=policy_learning_rate, inner_epoch=inner_epoch, 
                                                  eps_clip=epsilon_clip, max_step_count=max_step_length)
        
        if load_model is True:
            pickle_data = torch.load("./model/" + env_name + "_model.pl")
            alg_main.agent.policy_net_fast.load_state_dict(pickle_data["policy_net"])
            alg_main.agent.policy_net_fast.to(device)
            alg_main.agent.policy_net_slow.load_state_dict(pickle_data["policy_net"])
            alg_main.agent.policy_net_slow.to(device)
            alg_main.agent.critic_net_fast.load_state_dict(pickle_data["critic_net"])
            alg_main.agent.critic_net_fast.to(device)
            alg_main.agent.critic_net_slow.load_state_dict(pickle_data["critic_net"])
            alg_main.agent.critic_net_slow.to(device)

            alg_main.agent.policy_net_optimizer = torch.optim.Adam(alg_main.agent.policy_net_fast.parameters(), lr=policy_learning_rate)
            alg_main.agent.critic_net_optimizer = torch.optim.Adam(alg_main.agent.critic_net_fast.parameters(), lr=critic_learning_rate)
        
    elif algorithm_name == "trpo":
        max_kl_divergence = 0.0005
        alpha = 0.5
        alg_main = TrustRegionPolicyOptimziationMain(env=env, epoch=training_epoch, state_representation_size=state_representation_size, 
                                                     action_size=action_size, hidden_dim=hidden_dim, gamma=gamma, 
                                                     terminal_state_value=terminal_state_value, device=device,
                                                     critic_learning_rate=critic_learning_rate, max_kl_distance=max_kl_divergence, 
                                                     alpha=alpha, max_step_count=max_step_length)
        
        if load_model is True:
            pickle_data = torch.load("./model/" + env_name + "_model.pl")
            alg_main.agent.policy_net.load_state_dict(pickle_data["policy_net"])
            alg_main.agent.policy_net.to(device)
            alg_main.agent.critic_net.load_state_dict(pickle_data["critic_net"])
            alg_main.agent.critic_net.to(device)

            alg_main.agent.critic_net_optimizer = torch.optim.Adam(alg_main.agent.critic_net.parameters(), lr=critic_learning_rate)


    reward_list, count_list = alg_main.training()

    param_dict = {}
    param_dict["gamma"] = gamma
    param_dict["critic_learning_rate"] = critic_learning_rate
    param_dict["policy_learning_rate"] = policy_learning_rate
    param_dict["hidden_dim"] = hidden_dim
    param_dict["max_step_length"] = max_step_length
    param_dict["state_representation_size"] = state_representation_size
    param_dict["action_size"] = action_size

    if env_name == "stochatsic_negative_mdp" or env_name == "stochatsic_positive_mdp":
        param_dict["state_size"] = state_size
        param_dict["r_min"] = r_min
        param_dict["r_max"] = r_max
    
    if algorithm_name == "dqn":
        param_dict["buffer_size"] = buffer_size
        param_dict["batch_size"] = batch_size
        param_dict["explore_epsilon"] = explore_epsilon
        param_dict["target_change_num"] = target_change_num
        param_dict["exploration_decay_count"] = exploration_decay_count
    elif algorithm_name == "ppo":
        param_dict["inner_epoch"] = inner_epoch
        param_dict["epsilon_clip"] = epsilon_clip
    elif algorithm_name == "trpo":
        param_dict["max_kl_divergence"] = max_kl_divergence
        param_dict["alpha"] = alpha


    return reward_list, count_list, param_dict


def main():
    path_name = "./img"
    
    for env_name in ["cart_pole"]:
        for alg_name in ["dqn"]:
        # for alg_name in ["vpg", "a2c", "ppo", "dqn"]:
            training_epoch = 5000

            zero_reward_list = []
            non_zero_reward_list = []

            zero_count_list = []
            non_zero_count_list = []

            param_dict = None

            for random_seed in range(5):
                for zero_non_zero_flag in ["non_zero", "zero"]:
                    reward_list, count_list, param_dict = get_reward_list(random_seed=random_seed, env_name=env_name, 
                                                                          algorithm_name=alg_name, 
                                                                          zero_non_zero_flag=zero_non_zero_flag, 
                                                                          training_epoch=training_epoch)
                    
                    if zero_non_zero_flag == "zero":
                        zero_reward_list.append(reward_list)
                        zero_count_list.append(count_list)
                    else:
                        non_zero_reward_list.append(reward_list)
                        non_zero_count_list.append(count_list)

            pickle.dump(
                {
                    "zero_reward_list": zero_reward_list,
                    "zero_count_list": zero_count_list,
                    "non_zero_reward_list": non_zero_reward_list,
                    "non_zero_count_list": non_zero_count_list,
                    "param_dict": param_dict
                },
                open("./pickle_data/" + env_name + "_" + alg_name + "_result.pl", "wb")
            )

            plt_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
                                        env_name=env_name, alg_name=alg_name, file_path=path_name)
            
    
    # for env_name in ["mountain_car"]:
    #     for alg_name in ["ppo"]:
    #     # for alg_name in ["vpg", "a2c", "ppo", "dqn"]:
    #         if alg_name == "dqn":
    #             training_epoch = 5000
    #         else:
    #             training_epoch = 10000

    #         zero_reward_list = []
    #         non_zero_reward_list = []

    #         zero_count_list = []
    #         non_zero_count_list = []

    #         param_dict = None

    #         for random_seed in range(5):
    #             for zero_non_zero_flag in ["non_zero", "zero"]:
    #                 reward_list, count_list, param_dict = get_reward_list(random_seed=random_seed, env_name=env_name, 
    #                                                                       algorithm_name=alg_name, 
    #                                                                       zero_non_zero_flag=zero_non_zero_flag, 
    #                                                                       training_epoch=training_epoch, load_model=True)
                    
    #                 if zero_non_zero_flag == "zero":
    #                     zero_reward_list.append(reward_list)
    #                     zero_count_list.append(count_list)
    #                 else:
    #                     non_zero_reward_list.append(reward_list)
    #                     non_zero_count_list.append(count_list)

    #         pickle.dump(
    #             {
    #                 "zero_reward_list": zero_reward_list,
    #                 "zero_count_list": zero_count_list,
    #                 "non_zero_reward_list": non_zero_reward_list,
    #                 "non_zero_count_list": non_zero_count_list,
    #                 "param_dict": param_dict
    #             },
    #             open("./pickle_data/" + env_name + "_" + alg_name + "_result.pl", "wb")
    #         )

    #         plt_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
    #                                     env_name=env_name, alg_name=alg_name, file_path=path_name)
            
    
    
    # for env_name in ["stochatsic_positive_mdp"]:
    #     for alg_name in ["dqn"]:
    #     # for alg_name in ["vpg", "a2c", "ppo", "dqn"]:

    #         if alg_name in ["dqn", "ppo"]:
    #             training_epoch = 5000
    #         elif alg_name == "a2c":
    #             training_epoch = 20000
    #         elif alg_name == "vpg":
    #             training_epoch = 20000

    #         zero_reward_list = []
    #         non_zero_reward_list = []

    #         zero_count_list = []
    #         non_zero_count_list = []

    #         param_dict = None

    #         for random_seed in [0,1,2,]:
    #             for zero_non_zero_flag in ["non_zero", "zero"]:
    #                 reward_list, count_list, param_dict = get_reward_list(random_seed=random_seed, env_name=env_name, 
    #                                                                       algorithm_name=alg_name, 
    #                                                                       zero_non_zero_flag=zero_non_zero_flag, 
    #                                                                       training_epoch=training_epoch)
                    
    #                 if zero_non_zero_flag == "zero":
    #                     zero_reward_list.append(reward_list)
    #                     zero_count_list.append(count_list)
    #                 else:
    #                     non_zero_reward_list.append(reward_list)
    #                     non_zero_count_list.append(count_list)

    #         pickle.dump(
    #             {
    #                 "zero_reward_list": zero_reward_list,
    #                 "zero_count_list": zero_count_list,
    #                 "non_zero_reward_list": non_zero_reward_list,
    #                 "non_zero_count_list": non_zero_count_list,
    #                 "param_dict": param_dict
    #             },
    #             open("./pickle_data/" + env_name + "_" + alg_name + "_result.pl", "wb")
    #         )

    #         plt_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
    #                                     env_name=env_name, alg_name=alg_name, file_path=path_name)

    
    # for env_name in ["stochatsic_negative_mdp"]:
    #     for alg_name in ["vpg", "a2c", "ppo", "dqn"]:
    #         training_epoch = 5000

    #         zero_reward_list = []
    #         non_zero_reward_list = []

    #         zero_count_list = []
    #         non_zero_count_list = []

    #         param_dict = None

    #         for random_seed in range(5):
    #             for zero_non_zero_flag in ["non_zero", "zero"]:
    #                 reward_list, count_list, param_dict = get_reward_list(random_seed=random_seed, env_name=env_name, 
    #                                                                       algorithm_name=alg_name, 
    #                                                                       zero_non_zero_flag=zero_non_zero_flag, 
    #                                                                       training_epoch=training_epoch)
                    
    #                 if zero_non_zero_flag == "zero":
    #                     zero_reward_list.append(reward_list)
    #                     zero_count_list.append(count_list)
    #                 else:
    #                     non_zero_reward_list.append(reward_list)
    #                     non_zero_count_list.append(count_list)

    #         pickle.dump(
    #             {
    #                 "zero_reward_list": zero_reward_list,
    #                 "zero_count_list": zero_count_list,
    #                 "non_zero_reward_list": non_zero_reward_list,
    #                 "non_zero_count_list": non_zero_count_list,
    #                 "param_dict": param_dict
    #             },
    #             open("./pickle_data/" + env_name + "_" + alg_name + "_result.pl", "wb")
    #         )

    #         plt_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
    #                                     env_name=env_name, alg_name=alg_name, file_path=path_name)
            
            
    

if __name__ == "__main__":
    # result = pickle.load(open("./pickle_data/acrobot_dqn_result.pl", "rb"))
    main()
    
