from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCarCustomEnv(MountainCarEnv):
    def __init__(self, seed):
        super(MountainCarCustomEnv, self).__init__()
        self.seed = seed
        self.state_representation_size = 2
        self.action_size = 3

    def reset(self):
        state, _ = super(MountainCarCustomEnv, self).reset(seed=self.seed)
        return state

    def step(self, action):
        next_state, reward, done, _, _ = super(MountainCarCustomEnv, self).step(action)
        reward = -1
        return next_state, reward, done