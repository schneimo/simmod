import gym
from simmod.wrappers import UDRMujocoWrapper
from simmod.modification.mujoco import MujocoTextureModifier, MujocoMaterialModifier

if __name__ == '__main__':
    # Create the environment as you would normally do
    env = gym.make('FetchReach-v1')

    # Define modifier and algorithm for randomization
    # env.sim is the Mujoco simulation in the environment class
    tex_mod = MujocoTextureModifier(sim=env.sim)
    mat_mod = MujocoMaterialModifier(sim=env.sim)

    # Wrap the environment using the specific wrapper for the algorithm with the created modifier
    env = UDRMujocoWrapper(env, tex_mod, mat_mod)

    # Run algorithm and simulation as usual
    for _ in range(3):
        env.reset()
        for _ in range(100):
            env.step(0)
            env.render()
