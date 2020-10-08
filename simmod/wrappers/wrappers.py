import gym
from gym import Env

from simmod.algorithms import UniformDomainRandomization
from simmod.modification.mujoco.mujoco_modifier import MujocoBaseModifier


class UDRMujocoWrapper(gym.Wrapper):

    def __init__(self, env: Env, *modifiers: MujocoBaseModifier):
        super(UDRMujocoWrapper, self).__init__(env)
        assert env.unwrapped.sim is not None, "Assuming a Gym environment with a Mujoco simulation at variable 'sim'"
        self.sim = env.unwrapped.sim

        self.alg = UniformDomainRandomization(*modifiers)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

    def reset(self, **kwargs):
        self.alg.step()
        return self.env.reset(**kwargs)
