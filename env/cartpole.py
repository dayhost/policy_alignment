from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleCustomEnv(CartPoleEnv):
    def __init__(self, seed):
        super(CartPoleCustomEnv, self).__init__()
        self.seed = seed
        self.state_representation_size = 4
        self.action_size = 2

    def reset(self):
        state, _ = super(CartPoleCustomEnv, self).reset(seed=self.seed)
        return state

    def step(self, action):
        next_state, reward, done, _, _ = super(CartPoleCustomEnv, self).step(action)
        reward = +1
        return next_state, reward, done

