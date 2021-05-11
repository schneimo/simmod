import gym
import pybullet_envs
from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoTextureModifier, MujocoMaterialModifier, MujocoLightModifier

if __name__ == '__main__':
    # Create the environment as you would normally do
    env = gym.make('MinitaurBulletEnv-v0')

    env.unwrapped._pybullet_client.getDynamicsInfo()

    # Define modifier and algorithm for randomization
    # env.sim is the Mujoco simulation in the environment class

    # Run algorithm and simulation
    for _ in range(3):
        env.reset()
        for _ in range(100):
            env.step(0)
            env.render()
