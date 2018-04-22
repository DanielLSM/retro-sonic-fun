from retro_sonic_fun.common.utils import SonicDiscretizer
import numpy as np

ACTIONS = [
    ['LEFT'],  #0
    ['RIGHT'],  #1
    ['LEFT', 'DOWN'],  #2
    ['RIGHT', 'DOWN'],  #3
    ['DOWN'],  #4
    ['DOWN', 'B'],  #5
    ['B'],  #6
    ['RIGHT', 'B'],  #7
    ['LEFT', 'B']  #8
]

class Model(object):
    def __init__(self):
        pass

    def get_action(self, obs):
        raise NotImplementedError

class RandomModel(Model):
    def __init__(self, env, seed=42):
        self._action_controller = SonicDiscretizer(env)
        np.random.seed(seed)

    def get_action(self, obs):
        ac = np.random.randint(len(ACTIONS))
        return self._action_controller.action(ac)