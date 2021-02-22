import torch
import numpy as np
import copy
import gym

from algorithms.SAC import SAC
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths
from custom_envs.custom_pendulum import PendulumEnv
from custom_envs.custom_lunar_lander import LunarLanderContinuous
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
"""


#env = gym.make('HopperPyBulletEnv-v0')
#env_eval = gym.make('HopperPyBulletEnv-v0')

"""

"""

"""
"""

register( id='HopperPyBulletEnv-v1',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 0.75},
          max_episode_steps=1000,
          reward_threshold=2500.0)
register( id='HopperPyBulletEnv-v2',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 4.75},
          max_episode_steps=1000,
          reward_threshold=2500.0)



env1 = gym.make('HopperPyBulletEnv-v1')
env2 = gym.make('HopperPyBulletEnv-v2')
env_eval1 = gym.make('HopperPyBulletEnv-v1')
env_eval2 = gym.make('HopperPyBulletEnv-v2')

"""
""" """

register(
	id='HalfCheetahPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
    kwargs={'power':4.4},
	max_episode_steps=1000,
	reward_threshold=3000.0
	)

register(
	id='HalfCheetahPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
    kwargs={'power': 7.9},
	max_episode_steps=1000,
	reward_threshold=3000.0
	)
"""
register(
	id='HalfCheetahPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
    kwargs={'power': 2.9, "delta" : 0.0},
	max_episode_steps=1000,
	reward_threshold=3000.0
	)


register(
	id='HalfCheetahPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
    kwargs={'power': 2.9, "delta" : 40.0},
	max_episode_steps=1000,
	reward_threshold=3000.0
	)
	

env1 = gym.make('HalfCheetahPyBulletEnv-v1')
env2 = gym.make('HalfCheetahPyBulletEnv-v2')
env_eval1 = gym.make('HalfCheetahPyBulletEnv-v1')
env_eval2 = gym.make('HalfCheetahPyBulletEnv-v2')
"""
register(
	id='HumanoidPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.humanoid_env:HumanoidBulletEnv',
    kwargs={'power': 0.41},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HumanoidPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.humanoid_env:HumanoidBulletEnv',
    kwargs={'power': 1.41},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env1 = gym.make('HumanoidPyBulletEnv-v1')
env2 = gym.make('HumanoidPyBulletEnv-v2')
env_eval1 = gym.make('HumanoidPyBulletEnv-v1')
env_eval2 = gym.make('HumanoidPyBulletEnv-v2')


"""
env1 = gym.make('HalfCheetahPyBulletEnv-v1')
env2 = gym.make('HalfCheetahPyBulletEnv-v2')
env_eval1 = gym.make('HalfCheetahPyBulletEnv-v1')
env_eval2 = gym.make('HalfCheetahPyBulletEnv-v2')
"""
"""
register(
	id='Walker2DPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power':1.4},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='Walker2DPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power': 5.4},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)




env1 = gym.make('Walker2DPyBulletEnv-v1')
env2 = gym.make('Walker2DPyBulletEnv-v2')
env_eval1 = gym.make('Walker2DPyBulletEnv-v1')
env_eval2 = gym.make('Walker2DPyBulletEnv-v2')

"""

#env1 = LunarLanderContinuous(main_engine_power=13.0)
#env2 = LunarLanderContinuous(main_engine_power=3.0)
#env_eval1 = LunarLanderContinuous(main_engine_power=13.0)
#env_eval2 = LunarLanderContinuous(main_engine_power=3.0)
"""
register(
	id='InvertedPendulumPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.pendulum.inverted_pendulum_env:InvertedPendulumBulletEnv',
    kwargs={'torque_factor': 100, "gravity":9.8 },
	max_episode_steps=1000,
	reward_threshold=950.0,
	)

register(
	id='InvertedPendulumPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.pendulum.inverted_pendulum_env:InvertedPendulumBulletEnv',
    kwargs={'torque_factor': 100, "gravity": 90.8},
	max_episode_steps=1000,
	reward_threshold=950.0,
	)


env1 = gym.make('InvertedPendulumPyBulletEnv-v1')
env2 = gym.make('InvertedPendulumPyBulletEnv-v2')
env_eval1 = gym.make('InvertedPendulumPyBulletEnv-v1')
env_eval2 = gym.make('InvertedPendulumPyBulletEnv-v2')

"""
"""
register(
	id='AntPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.ant_env:AntBulletEnv',
    kwargs={'power': 2.5},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='AntPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.ant_env:AntBulletEnv',
    kwargs={'power': 6.5},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env1 = gym.make('AntPyBulletEnv-v1')
env2 = gym.make('AntPyBulletEnv-v2')
env_eval1 = gym.make('AntPyBulletEnv-v1')
env_eval2 = gym.make('AntPyBulletEnv-v2')

index = 4
register(
	id='InvertedPendulumSwingupPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.pendulum.inverted_pendulum_env:InvertedPendulumSwingupBulletEnv',
	max_episode_steps=1000,
    kwargs = {"length" : 1.0, "index":index},
	reward_threshold=950.0,
	)

register(
	id='InvertedPendulumSwingupPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.pendulum.inverted_pendulum_env:InvertedPendulumSwingupBulletEnv',
	max_episode_steps=1000,
    kwargs = {"length" : 2.5, "index":index+1},
	reward_threshold=950.0,
)

env1 = gym.make('InvertedPendulumSwingupPyBulletEnv-v1')
env2 = gym.make('InvertedPendulumSwingupPyBulletEnv-v2')
env_eval1 = gym.make('InvertedPendulumSwingupPyBulletEnv-v1')
env_eval2 = gym.make('InvertedPendulumSwingupPyBulletEnv-v2')

index = 0
register(
	id='Walker2DPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power': 0.40, "length" : 0.1, "index": index},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='Walker2DPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
    kwargs={'power': 0.40, "length" : 0.5, "index": index+1},
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

env1 = gym.make('Walker2DPyBulletEnv-v1')
env2 = gym.make('Walker2DPyBulletEnv-v2')
env_eval1 = gym.make('Walker2DPyBulletEnv-v1')
env_eval2 = gym.make('Walker2DPyBulletEnv-v2')

"""




index = 2
register( id='HopperPyBulletEnv-v1',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 0.75,  "thigh_length": 0.45, "leg_length" : 0.5, "leg_size" : 0.04, "foot_length" : 0.4, "thigh_size" :0.05, "index":index},
          max_episode_steps=1000,
          reward_threshold=2500.0)
register( id='HopperPyBulletEnv-v2',
          entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
          kwargs={'power': 0.75,  "thigh_length": 0.45, "leg_length" :2.5, "leg_size" : 0.04, "foot_length" : 0.4, "thigh_size" :0.05, "index":index+1},
          max_episode_steps=1000,
          reward_threshold=2500.0)

env1 = gym.make('HopperPyBulletEnv-v1')
env2 = gym.make('HopperPyBulletEnv-v2')
env_eval1 = gym.make('HopperPyBulletEnv-v1')
env_eval2 = gym.make('HopperPyBulletEnv-v2')
#seed = env.seed()[0]
#env_eval.seed(seed)


env = env1
env_eval = env_eval1

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
env1.reset()
env2.reset()
env_eval1.reset()
env_eval2.reset()

envs = [env1, env2]
x = 2000

print("x = " + str(x))
for i in range(200000):
    if i%1000 == 0:
        print("power = " + str(A.env.l_length))
        if i%x == 0:
            A.env = envs[0]
        else:
            A.env = envs[1]




    if i%1000 == 0:
        print(i)
    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        #A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
        pass
    if i%eval_interval==0:


        e = env_eval1
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
        print("reward at itr " + str(i) + " = " + str(rew_total)  + " power = " + str(e.l_length))#+ " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )


        e = env_eval2
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
        rew_total = rew_total / 10
        print("reward at itr " + str(i) + " = " + str(rew_total) + " power = " + str(e.l_length) ) # + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )

torch.save(A.replay_buffer, "mem")