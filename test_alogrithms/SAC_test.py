import torch
import numpy as np
import copy
import gym

from algorithms.SAC import SAC
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths
from custom_envs.custom_pendulum import PendulumEnv

import gym  # open ai gym


import custom_envs.pybulletgym_custom


#from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env import HopperBulletEnv

#env = gym.make("Pendulum-v0")
#env_eval = copy.deepcopy(env)
#env_eval = gym.make("Pendulum-v0")


#env = HopperBulletEnv()
#env_eval = HopperBulletEnv()

from gym.envs.registration import register
"""
register(
	id='Walker2DPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power': 7.40},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env = gym.make('Walker2DPyBulletEnv-v1')
env_eval = gym.make('Walker2DPyBulletEnv-v1')



register(
	id='AntPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.ant_env:AntBulletEnv',
    kwargs={'power': 3.5},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)



env = gym.make('Walker2DPyBulletEnv-v1')
env_eval = gym.make('Walker2DPyBulletEnv-v1')



register(
	id='AtlasPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.atlas_env:AtlasBulletEnv',
    kwargs={'power': 6.9},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)



env = gym.make('AtlasPyBulletEnv-v1')
env_eval = gym.make('AtlasPyBulletEnv-v1')



register(
	id='PusherPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.pusher_env:PusherBulletEnv',
    kwargs={'gravity': 9.81},
	max_episode_steps=150,
	reward_threshold=18.0,
)

env = gym.make('PusherPyBulletEnv-v1')
env_eval = gym.make('PusherPyBulletEnv-v1')


register(
	id='HalfCheetahPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
    kwargs={'power': 6.9},
	max_episode_steps=1000,
	reward_threshold=3000.0
	)

env = gym.make('HalfCheetahPyBulletEnv-v1')
env_eval = gym.make('HalfCheetahPyBulletEnv-v1')
"""

"""
"""

register(
	id='HumanoidPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.humanoid_env:HumanoidBulletEnv',
    kwargs={'power': 1.6},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env = gym.make('HumanoidPyBulletEnv-v1')
env_eval = gym.make('HumanoidPyBulletEnv-v1')



#env = gym.make('AntPyBulletEnv-v1')
#env_eval = gym.make('AntPyBulletEnv-v1')
#env.render() # call this before env.reset, if you want a window showing the environment
#env.reset()
"""
import pybulletgym

env = gym.make("HumanoidMuJoCoEnv-v0")
env_eval = gym.make("HumanoidMuJoCoEnv-v0")

"""


#env = gym.make('HopperPyBulletEnv-v0')
#env_eval = gym.make('HopperPyBulletEnv-v0')








seed = env.seed()[0]
env_eval.seed(seed)

env.l = 1.
env_eval.l  = 1.

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0003)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0003)

algo_nn_param = Algo_Param(gamma=0.99, alpha=0.2, tau=0.005, target_update_interval=1, automatic_alpha_tuning=True)


A = SAC(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=1000, memory_capacity=100000
        ,batch_size=512, alpha_lr=0.0003)
#A.load("q1", "q2", "q1", "q2", "policy_target")

save_interval = 2000
eval_interval = 2000

state = A.initalize()

for i in range(200000):
    if i%1000 == 0:
        print(i)
    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
        pass
    if i%eval_interval==0:

        e = env_eval
        e.render()

        rew_total = 0
        for _ in range(10):
            rew = 0
            s = e.reset()
            for j in range(A.max_episodes):
                a = A.get_action(s, evaluate=True)
                s, r, d, _ = e.step(a)

                rew += r

                if d == True:
                    break

            rew_total += rew
        rew_total = rew_total/10
        print("reward at itr " + str(i) + " = " + str(rew_total) )#+ " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )
torch.save(A.replay_buffer, "mem")