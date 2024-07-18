from gym.envs.classic_control.acrobot import AcrobotEnv


class AcrobotCustomEnv(AcrobotEnv):
    def __init__(self, seed):
        super(AcrobotCustomEnv, self).__init__()
        self.seed = seed
        self.state_representation_size = 6
        self.action_size = 3

    def reset(self):
        state, _ = super(AcrobotCustomEnv, self).reset(seed=self.seed)
        return state

    def step(self, action):
        next_state, reward, done, _, _ = super(AcrobotCustomEnv, self).step(action)
        reward = -1
        return next_state, reward, done