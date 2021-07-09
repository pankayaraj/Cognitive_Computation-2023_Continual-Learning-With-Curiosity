import torch
import numpy as np
import copy
import gym

from algorithms.SAC_task_relevance_policy import SAC_TR_P
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths
from custom_envs.custom_pendulum import PendulumEnv
from custom_envs.custom_lunar_lander import LunarLanderContinuous
import gym  # open ai gym
from custom_envs.sumo.custom_sumo_env import SUMOEnv

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
	id='Walker2DPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power': 0.40, "length" : 0.6},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env = gym.make('Walker2DPyBulletEnv-v1')
env_eval = gym.make('Walker2DPyBulletEnv-v1')



#env = gym.make('AtlasPyBulletEnv-v0')
#env_eval = gym.make('AtlasPyBulletEnv-v0')
"""



"""
register( id='HopperPyBulletEnv-v1',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 0.75, "thigh_length": 0.45, "leg_length" : 1.0, "foot_length" : 0.5, "leg_size" : 0.06 },
          max_episode_steps=1000,
          reward_threshold=2500.0)

register( id='HopperPyBulletEnv-v2',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 0.75, "thigh_length": 0.45, "leg_length" : 0.5, "foot_length" : 0.5, "index":1, "leg_size" : 0.04},
          max_episode_steps=1000,
          reward_threshold=2500.0)


env = gym.make('HopperPyBulletEnv-v1')
env_eval = gym.make('HopperPyBulletEnv-v1')
env_eval2 = gym.make('HopperPyBulletEnv-v2')
"""

env = PendulumEnv()
env_eval = PendulumEnv()
env_eval2 = PendulumEnv()


env.__init__()
env_eval.__init__()

env.l = 1.0
env_eval.l  = 1.0
mem_save_name = "old_buffers/Pendulum/l_1_0/mem"

#state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print((state_dim, action_dim))

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256,256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256,256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)

algo_nn_param = Algo_Param(gamma=0.99, alpha=0.2, tau=0.005, target_update_interval=1, automatic_alpha_tuning=True)

#Pendulum
A = SAC_TR_P(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=200, memory_capacity=10000
        ,batch_size=512, fifo_frac=0.05, alpha_lr=0.0003, env_type="roboschool", buffer_type="FIFO")

save_interval = 2000
eval_interval = 2000

state = A.initalize()

#A.load("q1", "q2", "q1", "q2", "policy_target")

eval = False
#eval = True
for i in range(30000):

    #if i%1000 == 0:
    #    print(i, A.alpha)

    if eval == False:
        A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        if eval == False:
            #A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
            pass
    if i%eval_interval==0:
        print("testing")
        e = env_eval


        rew_total = 0
        for _ in range(10):
            rew = 0

            #e.render()
            s = e.reset()

            for j in range(A.max_episodes):
                a = A.get_action(s, evaluate=True)
                s, r, d, _ = e.step(a)
                #e.render("gui")
                #if i%(save_interval*1) == 0:
                #    e.render(mode='rgb_array')
                rew += r

                if d == True:
                    break

            rew_total += rew
        rew_total = rew_total/10
        print("reward at itr " + str(i) + " = " + str(rew_total) + " length = " + str(e.l) )#+ " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )



torch.save(A.replay_buffer, mem_save_name)