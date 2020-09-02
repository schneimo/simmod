import gym
from gym import Env

from simmod.algorithms import UniformDomainRandomization
from simmod.modification.mujoco.mujoco_modifier import MujocoBaseModifier


class UDRMujocoWrapper(gym.Wrapper):

    def __init__(self, env: Env, *modifiers: MujocoBaseModifier, sim=None):
        super(UDRMujocoWrapper, self).__init__(env)
        if sim is None:
            self.sim = env.unwrapped.sim
        else:
            self.sim = sim
        self.alg = UniformDomainRandomization(*modifiers)

    def reset(self, **kwargs):
        self.alg.step()
        return self.env.reset(**kwargs)
