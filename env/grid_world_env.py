import numpy as np


class SimpleGridWorld(object):
    def __init__(self) -> None:

        self.state_size = 3
        self.action_size = 2

        state_list = list(range(self.state_size))
        self.state_representation = np.eye(self.state_size)[state_list]
        self.state_representation_size = self.state_size

    def reset(self):
        self.current_state = 0
        return self.state_representation[self.current_state]
        

    def step(self, action):
        if self.current_state == 0:
            if action == 0:
                next_state = 0
            elif action == 1:
                next_state = 1
            else:
                raise Exception("not support action: " + str(action))
        elif self.current_state == 1:
            if action == 0:
                next_state = 0
            elif action == 1:
                next_state = 2
            else:
                raise Exception("not support action: " + str(action))
        
        self.current_state = next_state
        
        if next_state == 0 and action == 0:
            reward = -1
        elif next_state == 0 and action == 1:
            reward = -1
        elif next_state == 1 and action == 0:
            reward = 0.95
        elif next_state == 1 and action == 1:
            reward = -1

        if next_state == 2:
            done = True
        else:
            done = False

        return self.state_representation[next_state], reward, done
        





