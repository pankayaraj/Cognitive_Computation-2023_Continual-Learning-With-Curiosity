from gym.envs.registration import register
import custom_envs.pybulletgym_custom
import gym

def make_array_env(change_variable, name):
    env, env_eval = [], []

    for i in range(len(change_variable)):
        if name == "HopperPyBulletEnv-v0":
            register(
                id='HopperPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('HopperPyBulletEnv-v' + str(i+1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i+1)))

            env[i].reset()
            env_eval[i].reset()

    return  env, env_eval



