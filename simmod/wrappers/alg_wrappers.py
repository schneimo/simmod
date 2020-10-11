import gym
from gym import Env

from simmod.algorithms import UniformDomainRandomization
from simmod.modification.mujoco.mujoco_modifier import MujocoBaseModifier
from simmod.common.parametrization import Execution


class UDRMujocoWrapper(gym.Wrapper):

    def __init__(self, env: Env, *modifiers: MujocoBaseModifier):
        super(UDRMujocoWrapper, self).__init__(env)
        assert env.unwrapped.sim is not None, "Assuming a Gym environment with a Mujoco simulation at variable 'sim'"
        self.sim = env.unwrapped.sim

        self.alg = UniformDomainRandomization(*modifiers)

    def step(self, action):
        action = self.alg.step(Execution.BEFORE_STEP, action=action)

        observation, reward, done, info = self.env.step(action)

        observation, reward, done, info = self.alg.step(execution=Execution.AFTER_STEP,
                                                        observation=observation, reward=reward, done=done, info=info)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.alg.step(Execution.RESET)
        return self.env.reset(**kwargs)
