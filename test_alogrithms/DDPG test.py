import torch
import numpy as np
import gym

from algorithms.DDPG import DDPG
from parameters import Algo_Param_DDPG, NN_Paramters, Load_Paths, Save_Paths
from test_alogrithms.test_other.normalized_env import NormalizedEnv
from test_alogrithms.test_other.normalized_env import NormalizedEnv

from custom_envs.custom_mountain_car import Continuous_MountainCarEnv
#env = NormalizedEnv(gym.make("Pendulum-v0"))
#env_eval = NormalizedEnv(gym.make("Pendulum-v0"))
"""
gym.register(
    id='SumoGUI-v0',
    entry_point='custom_envs.sumo:SUMOEnv_Initializer',
    kwargs={'port_no': 8871}
)
gym.register(
    id='SumoGUI-v1',
    entry_point='custom_envs.sumo:SUMOEnv_Initializer',
    kwargs={'port_no': 8872}
)

env = NormalizedEnv(gym.make('SumoGUI-v0'))
env_eval = NormalizedEnv(gym.make('SumoGUI-v1'))
"""
env = gym.make("BipedalWalker-v3")
env_eval = gym.make("BipedalWalker-v3")

gym.register(
	id='PusherPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.pusher_env:PusherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
)

gym.register(
	id='ReacherPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
    kwargs = {"torque_factor" : 0.05},
	max_episode_steps=150,
	reward_threshold=18.0,
	)


gym.register(
	id='StrikerPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.striker_env:StrikerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

gym.register(
	id='ThrowerPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.thrower_env:ThrowerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

env = gym.make("ReacherPyBulletEnv-v1")
env_eval = gym.make("ReacherPyBulletEnv-v1")




env = gym.make("StrikerPyBulletEnv-v1")
env_eval = gym.make("StrikerPyBulletEnv-v1")


env = gym.make('ThrowerPyBulletEnv-v1')
env_eval = gym.make('ThrowerPyBulletEnv-v1')

env = gym.make("ReacherPyBulletEnv-v1")
env_eval = gym.make("ReacherPyBulletEnv-v1")



env = gym.make("PusherPyBulletEnv-v1")
env_eval = gym.make("PusherPyBulletEnv-v1")
print(env.action_space)

#env = gym.make("MountainCarContinuous-v0")
#env_eval = gym.make("MountainCarContinuous-v0")

#env = Continuous_MountainCarEnv()
#env_eval = Continuous_MountainCarEnv()

#state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#[512, 512, 256, 64]
q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[400, 300],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0005)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[400, 300],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)

algo_nn_param = Algo_Param_DDPG(gamma=0.99, tau=0.0001, target_update_interval=1, noise="Ornstein", std=0.01)


A = DDPG(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=1000, memory_capacity=100000
        ,batch_size=64, noise_type ="Ornstein", ou_theta = 0.15, ou_sigma = 0.2,  ou_mu = 0.0, env_type="roboschool")


save_interval = 2000
eval_interval = 2000

state = A.initalize()
#A.load("q", "q_target", "policy", "policy_target")
for i in range(200000):

    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        A.save("q", "q_target", "policy", "policy_target")
        pass
    if i%eval_interval==0:
        #print(A.env.speed)
        e = env_eval
        e.render()

        rew_total = 0
        for j in range(10):
            s = e.reset()
            rew = 0
            a_ = 0
            for j in range(A.max_episodes):
                a = A.get_action(s, evaluate=True)
                a_ += a
                s, r, d, _ = e.step(a)
                rew += r
                #e.render("gui")

                if d == True:
                    break
            rew_total += rew
        #print(a_/j)
        print("reward at itr " + str(i) + " = " + str(rew_total/10)  )
torch.save(A.replay_buffer, "mem")