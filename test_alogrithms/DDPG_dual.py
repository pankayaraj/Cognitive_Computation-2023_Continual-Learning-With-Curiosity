import torch
import numpy as np
import gym

from algorithms.DDPG import DDPG
from parameters import Algo_Param_DDPG, NN_Paramters, Load_Paths, Save_Paths
from test_alogrithms.test_other.normalized_env import NormalizedEnv
from test_alogrithms.test_other.normalized_env import NormalizedEnv

from custom_envs.custom_mountain_car import Continuous_MountainCarEnv

gym.register(
	id='ReacherPyBulletEnv-v1',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
    kwargs = {"torque_factor" : 0.05},
	max_episode_steps=150,
	reward_threshold=18.0,
	)

gym.register(
	id='ReacherPyBulletEnv-v2',
	entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
    kwargs = {"torque_factor" : 50},
	max_episode_steps=150,
	reward_threshold=18.0,
	)

env1 = gym.make("ReacherPyBulletEnv-v1")
env_eval1 = gym.make("ReacherPyBulletEnv-v1")
env2 = gym.make("ReacherPyBulletEnv-v2")
env_eval2 = gym.make("ReacherPyBulletEnv-v2")



env = env1
env_eval = env_eval1


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#[512, 512, 256, 64]
q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[400, 300],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.001)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[400, 300],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)

algo_nn_param = Algo_Param_DDPG(gamma=0.99, tau=0.001, target_update_interval=1, noise="Ornstein", std=0.01)


A = DDPG(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=1000, memory_capacity=100000
        ,batch_size=32, noise_type ="Ornstein", ou_theta = 0.15, ou_sigma = 0.2,  ou_mu = 0.0, env_type="roboschool")


save_interval = 2000
eval_interval = 2000

state = A.initalize()

env1.reset()
env2.reset()

envs = [env1, env2]
x = 2000

print("x = " + str(x))
for i in range(200000):

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)

    A.update()

    if i%1000 == 0:
        print("power = " + str(A.env.torque_factor))
        if i%x == 0:
            A.env = envs[1]
        else:
            A.env = envs[0]

    if i%1000 == 0:
        print(i)

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        A.save("q", "q_target", "policy", "policy_target")
        pass
    if i%eval_interval==0:


        e = env_eval1
        #e.render()

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
        print("reward at itr " + str(i) + " = " + str(rew_total)  + " power = " + str(e.torque_factor))#+ " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )


        e = env_eval2
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
        print("reward at itr " + str(i) + " = " + str(rew_total) + " power = " + str(e.torque_factor) ) # + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )

torch.save(A.replay_buffer, "mem")