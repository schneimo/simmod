import gym
from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoTextureModifier, MujocoMaterialModifier, MujocoLightModifier

if __name__ == '__main__':
    # Create the environment as you would normally do
    env = gym.make('HandReach-v0')

    # Define modifier and algorithm for randomization
    # env.sim is the Mujoco simulation in the environment class
    #mod_tex = MujocoTextureModifier(sim=env.sim)
    mod_mat = MujocoMaterialModifier(sim=env.sim)
    env = UDRMujocoWrapper(env, mod_mat)

    # Run algorithm and simulation
    env.reset()
    for _ in range(100):
        env.step(0)
        env.render()
