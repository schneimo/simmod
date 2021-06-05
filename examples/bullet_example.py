import gym
import pybullet_envs

if __name__ == '__main__':
    # Create the environment as you would normally do
    env = gym.make('MinitaurBulletEnv-v0')

    # Run algorithm and simulation as usual
    for _ in range(3):
        env.reset()
        for _ in range(100):
            env.step(0)
            env.render()
