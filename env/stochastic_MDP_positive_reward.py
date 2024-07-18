import numpy as np
import itertools
import copy


class StochasticPositiveMDP(object):
    def __init__(self, state_size, terminal_state_size, action_size, random_seed, r_min, r_max) -> None:
        # set random seed
        np.random.seed(random_seed)
        # set param
        self.state_size = state_size
        self.terminal_state_size = terminal_state_size
        self.action_size = action_size

        # use one-hot encoding to make state representation
        state_list = list(range(self.state_size))
        self.state_representation = np.eye(self.state_size)[state_list]
        self.state_representation_size = self.state_size

        self.non_terminal_state_list = list(range(state_size - terminal_state_size))
        self.terminal_state_list = list(range(state_size - terminal_state_size, state_size))

        self.r_min = r_min
        self.r_max = r_max

        self._get_transition_probability()
        self._get_reward_function()

    def _get_transition_probability(self):
        # generate random matrix
        non_terminal_state_size =self.state_size - self.terminal_state_size
        non_terminal_transition_probability = np.random.uniform(
            low=1, high=2, size=(non_terminal_state_size, non_terminal_state_size, self.action_size))
        non_terminal_2_terminal_transition_probability = np.random.uniform(
            low=1, high=2, size=[non_terminal_state_size, self.terminal_state_size, self.action_size])
        identity_matrix = np.concatenate([np.eye(self.terminal_state_size).reshape([self.terminal_state_size, self.terminal_state_size, 1])] * self.action_size, axis=2)
        zero_matrix = np.zeros(shape=[self.terminal_state_size, non_terminal_state_size, self.action_size])

        # process non_terminal_2_terminal_transition_probability
        # construct a policy that cannot reach terminal state
        # 构建一个无法走到terminal state value的策略，在reward function为正的时候，只有这个策略是最优的
        choose_action_list = list(np.random.choice(self.action_size, non_terminal_state_size))
        for choose_state_i, choose_action in zip(range(non_terminal_state_size), choose_action_list):
            for choose_state_j in range(self.terminal_state_size): 
                non_terminal_2_terminal_transition_probability[choose_state_i][choose_state_j][choose_action] = 0
        
        # concate the transition probability together
        non_terminal_state_transtion_matrix = np.concatenate([non_terminal_transition_probability, non_terminal_2_terminal_transition_probability], axis=1)
        terminal_state_transition_matrix = np.concatenate([zero_matrix, identity_matrix], axis=1)
        
        # normalization
        for choose_state_i in range(non_terminal_state_size):
            for action_idx in range(self.action_size):
                temp_vec = non_terminal_state_transtion_matrix[choose_state_i, :, action_idx]
                non_terminal_state_transtion_matrix[choose_state_i, :, action_idx] = temp_vec / np.sum(temp_vec)
        
        self.transition_probability = np.concatenate([non_terminal_state_transtion_matrix, terminal_state_transition_matrix], axis=0)


    def _get_reward_function(self):
        # generate random reward function
        non_terminal_state_size =self.state_size - self.terminal_state_size
        non_terminal_reward_func = np.random.uniform(
            low=self.r_min, high=self.r_min + 0.1, size=(non_terminal_state_size, non_terminal_state_size, self.action_size))
        non_terminal_2_terminal_reward_func = np.random.uniform(
            low=self.r_max - 0.1, high=self.r_max, size=[non_terminal_state_size, self.terminal_state_size, self.action_size])
        non_terminal_state_reward_func = np.concatenate([non_terminal_reward_func, non_terminal_2_terminal_reward_func], axis=1)
        
        # # generate random reward function
        # non_terminal_state_reward_func = np.random.uniform(
        #     low=self.r_min, high=self.r_max, size=(self.state_size, self.state_size, self.action_size))
        
        self.reward_function = non_terminal_state_reward_func

    def reset(self):
        self.current_state = 0
        return self.state_representation[self.current_state]

    def step(self, action):
        
        transition_state_vec = list(self.transition_probability[self.current_state, :, action])
        next_state = np.random.choice(self.state_size, 1, p=transition_state_vec)[0]
        reward = self.reward_function[self.current_state, next_state, action]

        next_state_obs = self.state_representation[next_state]
        self.current_state = next_state

        if self.current_state in self.terminal_state_list:
            done = True
        else:
            done = False

        return next_state_obs, reward, done

    def stop(self):
        pass

    def get_minimal_positive_transition_probability(self):

        action_list = [list(range(self.action_size))] * (self.state_size - self.terminal_state_size)

        minimal_positive_transition_probability = 10

        for choosed_policy in itertools.product(*action_list):
            # choose a policy
            # print("action is: " + str(choosed_policy))
            choosed_policy_list = copy.deepcopy(list(choosed_policy))
            for _ in range(self.terminal_state_size):
                choosed_policy_list.append(0)
            
            # construct the transition probability matrix based on the policy
            transition_vec_list = []
            for current_state in range(self.state_size):
                transition_vec_list.append(copy.deepcopy(self.transition_probability[current_state, :, [choosed_policy_list[current_state]]]))  
            transiton_matrix_4_current_policy = np.concatenate(transition_vec_list, axis=0)
            # print(transiton_matrix_4_current_policy)
            
            # construct the power of transition matrix and absorption probability
            probability_matrix_exp = np.linalg.matrix_power(transiton_matrix_4_current_policy, self.state_size - 1)
            # print(probability_matrix_exp)
            probaility_matrix_for_terminal_state = probability_matrix_exp[:self.state_size - self.terminal_state_size, 
                                                                          self.state_size - self.terminal_state_size:]
            # print(probaility_matrix_for_terminal_state)

            # find the minimum positive absorption probability
            if len(np.nonzero(probaility_matrix_for_terminal_state)[0]) == 0 and len(np.nonzero(probaility_matrix_for_terminal_state)[1]) == 0:
                continue

            current_positive_min = np.min(probaility_matrix_for_terminal_state[np.nonzero(probaility_matrix_for_terminal_state)])
            if  current_positive_min < minimal_positive_transition_probability:
                minimal_positive_transition_probability = current_positive_min

        return minimal_positive_transition_probability


# state_size = 3
# terminal_state_size = 1
# action_size = 2
# random_seed = 100
# low_reward = 0.5
# high_reward = 1
# mdp = StochasticMDP(state_size, terminal_state_size, action_size, low_reward, high_reward, random_seed)
# a = mdp.get_minimal_positive_transition_probability()
# print(a)
# mdp.reset()
# done = False
# for _ in range(1000):
#     print(mdp.current_state)
#     np.random.seed(1)
#     action = np.random.randint(2)
#     _, _, done = mdp.step(action=action)
#     print(mdp.current_state)
#     if done:
#         break
# if done is False:
#     print("Not Done")

