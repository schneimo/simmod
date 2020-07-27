import gym
from gym.envs.mujoco import MujocoEnv


class UDRMujocoWrapper(gym.Wrapper):

    def __init__(self, env: MujocoEnv):
        super(UDRMujocoWrapper, self).__init__(env)
        self.sim = env.sim

    def step(self, action: float):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
