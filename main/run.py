import torch
import numpy as np
from itertools import count

import argparse

import custom_envs.custom_cartpole
from algorithms.SAC import SAC
from algorithms.SAC_w_Curiosity import SAC_with_Curiosity
from algorithms.SAC_w_Reward_based_Curiosity import SAC_with_reward_based_Curiosity

from algorithms.SAC_w_Cur_Buffer import SAC_with_Curiosity_Buffer
from algorithms.SAC_w_IRM_Cur_buffer import SAC_with_IRM_Curiosity_Buffer

from algorithms.SAC_test import SAC_Test
from algorithms.DDPG import DDPG
from algorithms.DDPG_w_Cur_Buffer import DDPG_with_Curiosity_Buffer

from algorithms.q_learning import Q_learning
from algorithms.q_learning_cur_buffer import Q_learning_w_cur_buf

from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths, Algo_Param_DDPG

from custom_envs.custom_pendulum import PendulumEnv
from custom_envs.custom_acrobat import AcrobotEnv
from custom_envs.custom_cartpole import CartPoleEnv

from util.roboschool_util.make_new_env import make_array_env
from util.roboschool_util.make_env_schedule import sine_schedule_generator, spurcious_fluxuation_generator, linear_schedule_generator

from set_arguments import set_arguments


parser = argparse.ArgumentParser(description='SAC arguments')

#SUMMARY
#SAC + Half_Reservior_FIFO_with_FT = HRF
#SAC + FIFO = FIFO
#SAC + MRT = MTR
#SAC_w_cur_buffer + + Half_Reservior_FIFO_with_FT = Reservoir with task seperation

#ALGORITHMS
#"SAC_w_cur_buffer"
#"DDPG_w_cur_buffer"
#SAC
#DDPG
#"Q_Learning"
#"Q_Learning_w_cur_buffer"


#BUFFERS
#FIFO
#"Half_Reservior_FIFO_with_FT"
#"MTR"


#not done yet
#"FIFO_FT"

#ENVIORNMENTS
#"Pendulum-v0"
#"HopperPyBulletEnv-v0"
#"Walker2DPyBulletEnv-v0"
#"Walker2DPyBulletEnv-v0_leg_len"
#"AntPyBulletEnv-v0"
#'AtlasPyBulletEnv-v0'
#"HumanoidPyBulletEnv-v0"
#HalfCheetahPyBulletEnv-v0g
#"ReacherPyBulletEnv-v0"
#"InvertedPendulumSwingupPyBulletEnv-v0
#Cartpole-v0
#Acrobat


#"discrete"
#"sine_flux"
#"spur_flux"
#"linear"

#"TS_C_HRF"
#"MTR_high_20"
#"HCRRF"

#these two supersedes other argumetns and makes the relevant choices for hyperparamter as it appears in the paper
#set superceding to false if you want to set custom arguments


parser.add_argument("--superseding", type=bool, default=True)
parser.add_argument("--env_change_type", type=str, default="sine_flux")
parser.add_argument("--supersede_env", type=str, default="Pendulum")
parser.add_argument("--supersede_buff", type=str, default="HRF")

parser.add_argument("--algo", type=str, default="SAC_w_cur_buffer")
parser.add_argument("--buffer_type", type=str, default="Custom")
parser.add_argument("--env", type=str, default="HopperPyBulletEnv-v0")
parser.add_argument("--env_type", type=str, default="roboschool")
"""
"""
"""
parser.add_argument("--algo", type=str, default="SAC_w_cur_buffer")
parser.add_argument("--buffer_type", type=str, default="Custom")
parser.add_argument("--env", type=str, default="Pendulum-v0")
parser.add_argument("--env_type", type=str, default="classic_control")
"""

#for sine flux

parser.add_argument("--min_lim", type=float, default=1.0)
parser.add_argument("--max_lim", type=float, default=1.8)
parser.add_argument("--factor", type=float, default=0.0001)
parser.add_argument("--sche_steps", type=float, default=400000)

"""
parser.add_argument("--algo", type=str, default="Q_Learning")
parser.add_argument("--buffer_type", type=str, default="MTR")
parser.add_argument("--env", type=str, default="Cartpole-v0")
parser.add_argument("--env_type", type=str, default="classic_control")
"""
#IRM parameters
parser.add_argument("--do_irm", type=bool, default=False)
parser.add_argument("--apply_irm_on_policy", type=bool, default=True)
parser.add_argument("--apply_irm_on_critic", type=bool, default=False)
parser.add_argument("--irm_coefficient_p", type=float, default=0.1)
parser.add_argument("--irm_coefficient_q", type=float, default=0.0005)

parser.add_argument("--load_from_old", type=bool, default=False)
parser.add_argument("--load_index", type=int, default=3) #to indicate which change of varaiable we are at
parser.add_argument("--starting_time_step", type=int, default=0) #from which time fram to start things

parser.add_argument("--experiment_no", type=int, default=7)


#parser.add_argument("--fifo_frac", type=float, default=0.34)
parser.add_argument("--fifo_frac", type=float, default=0.05)

parser.add_argument("--no_curiosity_networks", type=int, default=0)
#parser.add_argument("--no_curiosity_networks", type=int, default=1)
#parser.add_argument("--no_curiosity_networks", type=int, default=3)


parser.add_argument("--init_cur_at_task_change", type=bool, default=False)
parser.add_argument("--init_alpha_at_task_change", type=bool, default=False)


parser.add_argument("--policy", type=str, default="gaussian")

parser.add_argument("--hidden_layers", type=list, default=[256, 256])
#parser.add_argument("--hidden_layers", type=list, default=[64, 64])

parser.add_argument("--lr", type=float, default=0.0003)
#parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--automatic_entropy_tuning", type=bool, default=True)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=40000)
parser.add_argument("--eval-interval", type=int, default=2000)
parser.add_argument("--restart_alpha", type=bool, default=False)
parser.add_argument("--restart_alpha_interval", type=int, default=50000)

parser.add_argument("--batch_size", type=int, default=512)
#parser.add_argument("--batch_size", type=int, default=32)
#parser.add_argument("--batch_size", type=int, default=128)

#parser.add_argument("--memory_size", type=int, default=40000) #walker
#parser.add_argument("--memory_size", type=int, default=100000) #walker
#parser.add_argument("--memory_size", type=int, default=20000)
#parser.add_argument("--memory_size", type=int, default=20000)
parser.add_argument("--memory_size", type=int, default=50000)

#parser.add_argument("--no_steps", type=int, default=180000)
#parser.add_argument("--no_steps", type=int, default=150000)
#parser.add_argument("--no_steps", type=int, default=360000)
#parser.add_argument("--no_steps", type=int, default=230000)
parser.add_argument("--no_steps", type=int, default=400000)


parser.add_argument("--max_episodes", type=int, default=1000)
#parser.add_argument("--max_episodes", type=int, default=200)
#parser.add_argument("--max_episodes", type=int, default=1000)

parser.add_argument("--save_directory", type=str, default="models/native_SAC_catastropic_forgetting/diff_length")

parser.add_argument("--n_k", type=int, default=500)
parser.add_argument("--l_k", type=int, default=30000)
parser.add_argument("--m_f", type=int, default=2.5)

parser.add_argument("--fow_cur_weight", type=float, default=0.0)
parser.add_argument("--inv_cur_weight", type=float, default=1.0)
parser.add_argument("--rew_cur_weight", type=float, default=0.0)

parser.add_argument("--priority", type=str, default="uniform")
#parser.add_argument("--priority", type=str, default="curiosity")


parser.add_argument("--mtr_buff_no", type=int, default=3)

parser.add_argument("--save_buff_after", type=int, default=-1)


c = 0
args = parser.parse_args()



if args.superseding == True:
    args, change_varaiable_at, change_varaiable = set_arguments(args)

if args.env_type == "roboschool":
    if args.env == "Walker2DPyBulletEnv-v0":
        if args.env_change_type == "spur_flux":
            change_varaiable_at = [0, 86067, 86067 + 25000, 150939, 150939+ 12500, 217851, 217851 + 12500, 282606, 282606 + 12500, 315080,  315080 + 12500, 376422, 376422 + 12500]
            change_varaiable = [1.4, 7.4, 1.4, 13.4,  1.4, 7.4, 1.4, 7.4, 1.4, 13.4,  1.4, 13.4, 1.4,]
            change_varaiable_test = [1.4, 7.4, 13.4]


    elif args.env == "HopperPyBulletEnv-v0":

        #change_varaiable_at = [0, 86067, 86067 + 25000, 150939, 150939 + 12500, 217851, 217851 + 12500, 282606,
        #                           282606 + 12500, 315080, 315080 + 12500, 376422, 376422 + 12500]
        #change_varaiable = [4.75, 0.75, 4.75, 8.75, 4.75, 0.75, 4.75, 0.75, 4.75, 8.75, 4.75, 8.75, 4.75,]
        change_varaiable_test = [0.75, 4.75, 8.75]
        #change_varaiable_at = [ 0, 86067, 86067 + 12500, 150939, 150939 + 12500, 217851, 217851 + 12500, 282606,
        #                                              282606 + 12500, 315080, 315080 + 12500, 376422, 376422 + 12500]
        #change_varaiable_at = [0, 86067, 86067 + 5000, 150939, 150939 + 5000, 217851, 217851 + 5000, 282606,
        #                       282606 + 5000, 315080, 315080 + 5000, 376422, 376422 + 5000]
        #change_varaiable = [4.75, 0.75, 4.75, 8.75, 4.75, 0.75, 4.75, 0.75, 4.75, 8.75, 4.75, 4.75, 0.75, ]
        """
        x = []
        j = 0
        for i in range(400000):

            print(j, i)
            if j == 0:
                x.append(change_varaiable[j])
                if i > change_varaiable_at[j]:
                    j += 1

            elif j == len(change_varaiable_at)-1:
                x.append(change_varaiable[j])
            else:
                x.append(change_varaiable[j])
                if i == change_varaiable_at[j]:
                    j += 1

        import matplotlib.pyplot as plt
        plt.plot(x)
        plt.show()
        """

        change_varaiable_at = [0 for i in range(20)]
        change_varaiable = []
        j = 0
        for i in range(20):
            j = (j + 1) % 2
            change_varaiable_at[i] = i * 20000
            if j == 0:
                change_varaiable.append(change_varaiable_test[0])
            elif j == 1:
                change_varaiable.append(change_varaiable_test[2])
        t = 0
        for i in range(20):
            change_varaiable_at.insert(t+1, change_varaiable_at[t] + 2000)
            change_varaiable.insert(t+1, change_varaiable_test[1])
            t += 2

        """
        """
print(change_varaiable)
#only for flux settings
if args.env_change_type == "sine_flux" and args.env == "Pendulum-v0":
    schedule_generator = sine_schedule_generator(min_limit=args.min_lim, max_limit=args.max_lim, factor=args.factor, no_iteration=args.sche_steps )
elif args.env_change_type == "sine_flux" and args.env == "HopperPyBulletEnv-v0":
    schedule_generator = sine_schedule_generator(min_limit=args.min_lim, max_limit=args.max_lim, factor=args.factor, no_iteration=args.sche_steps )
elif args.env_change_type == "spur_flux":
    schedule_generator = spurcious_fluxuation_generator(min_limit=args.min_lim, max_limit=args.max_lim, no_iteration=args.no_steps)
elif args.env_change_type == "linear":
    schedule_generator = linear_schedule_generator(min_limit=args.min_lim, max_limit=args.max_lim, no_iteration=args.no_steps)


print(args.algo + " , " + args.buffer_type + " , " + args.priority)
if args.env_type == "classic_control":
    if args.env == "Pendulum-v0":
        env = PendulumEnv()
        env_eval = PendulumEnv()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        ini_env = env

    elif args.env == "Acrobat-v0":
        env = AcrobotEnv()
        env_eval = AcrobotEnv()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        ini_env = env

    elif args.env == "Cartpole-v0" or args.env == "Cartpole-v0_masspole":
        env = CartPoleEnv()
        env_eval = CartPoleEnv()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        ini_env = env

elif args.env_type == "roboschool":

    if args.env_change_type == "sine_flux":
        change_varaiable = [0.75]
        change_varaiable_test = [0.75, 4.75, 8.75]
        env, env_eval = make_array_env(change_varaiable, args.env, args.env_change_type, change_varaiable_test)

        state_dim = env[0].observation_space.shape[0]
        action_dim = env[0].action_space.shape[0]

        ini_env = env[0]

        for e in env:
            print(e.power)
        for e in env_eval:
            print(e.power)
    else:
        if args.env == "Walker2DPyBulletEnv-v0" or args.env == "HopperPyBulletEnv-v0":
            if args.env_change_type == "spur_flux":
                env, env_eval = make_array_env(change_varaiable, args.env, args.env_change_type, change_varaiable_test)
            else:
                env, env_eval = make_array_env(change_varaiable, args.env, args.env_change_type)
        else:
            env, env_eval = make_array_env(change_varaiable, args.env, args.env_change_type)

        state_dim = env[0].observation_space.shape[0]
        action_dim = env[0].action_space.shape[0]

        ini_env = env[0]



if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

hidden_layers = args.hidden_layers

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=args.hidden_layers,
                          non_linearity=torch.relu, device=device, l_r=args.lr)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=args.hidden_layers,
                          non_linearity=torch.relu, device=device, l_r=args.lr)
icm_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=args.hidden_layers,
                          non_linearity=torch.relu, device=device, l_r=args.lr)

if args.algo == "DDPG_w_cur_buffer" or args.algo == "DDPG":
    algo_nn_param = Algo_Param_DDPG(gamma=args.gamma, tau=args.tau,
                                    target_update_interval=args.target_update_interval,
                noise = "gaussian", depsilon = 50000, std = 1.0)

else:
    algo_nn_param = Algo_Param(gamma=args.gamma, alpha=args.alpha, tau=args.tau,
                               target_update_interval=args.target_update_interval,
                               automatic_alpha_tuning=args.automatic_entropy_tuning)

buffer_type = args.buffer_type

if args.algo == "SAC":
    A = SAC(ini_env, q_nn_param, policy_nn_param, algo_nn_param,
        max_episodes=args.max_episodes, memory_capacity=args.memory_size,
        batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type, fifo_frac=args.fifo_frac,
            change_at=change_varaiable_at[1:], mtr_buff_no=args.mtr_buff_no )
#elif args.algo == "SAC_w_cur":
#    A = SAC_with_Curiosity(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=args.max_episodes,
#                           memory_capacity=args.memory_size
#                           , batch_size=args.batch_size, alpha_lr=args.lr,
#                           )
#elif args.algo == "SAC_w_r_cur":
#    A = SAC_with_reward_based_Curiosity(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=args.max_episodes,
#                           memory_capacity=args.memory_size
#                           , batch_size=args.batch_size, alpha_lr=args.lr)
elif args.algo == "SAC_w_cur_buffer":
    if not args.do_irm:
        A = SAC_with_Curiosity_Buffer(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param,
                               max_episodes=args.max_episodes,
                               memory_capacity=args.memory_size
                               , batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type, fifo_frac=args.fifo_frac
                                      , no_cur_network=args.no_curiosity_networks,
                                      reset_cur_on_task_change=args.init_cur_at_task_change,  reset_alpha_on_task_change=args.init_alpha_at_task_change,
                                      fow_cur_w=args.fow_cur_weight, inv_cur_w=args.inv_cur_weight, rew_cur_w=args.rew_cur_weight,
                                      n_k=args.n_k, l_k=args.l_k, m_k=args.m_f,
                                      priority = args.priority)

    else:
        A = SAC_with_IRM_Curiosity_Buffer(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param,
                                      max_episodes=args.max_episodes,
                                      memory_capacity=args.memory_size
                                      , batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type,
                                      fifo_frac=args.fifo_frac
                                      , no_cur_network=args.no_curiosity_networks,
                                      reset_cur_on_task_change=args.init_cur_at_task_change,
                                      reset_alpha_on_task_change=args.init_alpha_at_task_change,
                                      fow_cur_w=args.fow_cur_weight, inv_cur_w=args.inv_cur_weight,
                                      rew_cur_w=args.rew_cur_weight,
                                      n_k=args.n_k, l_k=args.l_k, m_k=args.m_f,
                                      priority=args.priority,
                                      irm_coff_policy=args.irm_coefficient_p, irm_coff_critic=args.irm_coefficient_q,
                                      irm_on_policy=args.apply_irm_on_policy, irm_on_critic=args.apply_irm_on_critic,
                                        change_at=change_varaiable_at[1:])


elif args.algo == "SAC_test":
    A = SAC_Test(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param,
                           max_episodes=args.max_episodes,
                           memory_capacity=args.memory_size
                           , batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type, fifo_frac=args.fifo_frac
                                  , no_cur_network=args.no_curiosity_networks,
                            change_at=change_varaiable_at[1:])

elif args.algo == "Q_Learning":
    A = Q_learning(ini_env, q_nn_param=q_nn_param, algo_param=algo_nn_param, max_episodes=args.max_episodes, memory_capacity=args.memory_size
                           , batch_size=args.batch_size, buffer_type=buffer_type, fifo_frac=args.fifo_frac, change_at=change_varaiable_at[1:]
                   , mtr_buff_no=args.mtr_buff_no)
elif args.algo == "Q_Learning_w_cur_buffer":

    A = Q_learning_w_cur_buf(ini_env, q_nn_param=q_nn_param,icm_nn_param=icm_nn_param, algo_param=algo_nn_param,
                             max_episodes=args.max_episodes,memory_capacity=args.memory_size, batch_size=args.batch_size,
                             buffer_type=buffer_type, fifo_frac=args.fifo_frac
                             , no_cur_network=args.no_curiosity_networks,
                             reset_cur_on_task_change=args.init_cur_at_task_change,
                             reset_alpha_on_task_change=args.init_alpha_at_task_change,
                             fow_cur_w=args.fow_cur_weight, inv_cur_w=args.inv_cur_weight, rew_cur_w=args.rew_cur_weight,
                             n_k=args.n_k, l_k=args.l_k, m_k=args.m_f,
                             priority=args.priority
                             )
elif args.algo == "DDPG":
    A = DDPG(ini_env, q_nn_param=q_nn_param, policy_nn_param=policy_nn_param, algo_nn_param=algo_nn_param,
             max_episodes=args.max_episodes, memory_capacity=args.memory_size
        ,batch_size=args.batch_size, buffer_type=buffer_type,
             noise_type ="Ornstein", ou_theta = 0.15, ou_sigma = 0.2,  ou_mu = 0.0, sigma_min=None, anneal_epsilon=False,
             env_type="roboschool", fifo_frac=args.fifo_frac, change_at=change_varaiable_at[1:], reset_alpha_on_task_change=args.init_alpha_at_task_change
             )
elif args.algo == "DDPG_w_cur_buffer":
    A = DDPG_with_Curiosity_Buffer(ini_env, q_nn_param=q_nn_param, policy_nn_param=policy_nn_param, algo_nn_param=algo_nn_param, icm_nn_param=icm_nn_param,
             max_episodes=args.max_episodes, memory_capacity=args.memory_size, batch_size=args.batch_size, buffer_type=buffer_type,
             noise_type="Ornstein", ou_theta=0.15, ou_sigma=0.3, ou_mu=0.0, sigma_min=None, anneal_epsilon=False,
            env_type="roboschool", fifo_frac=args.fifo_frac, change_at=change_varaiable_at[1:],
                                   no_cur_network=args.no_curiosity_networks,
                                  reset_cur_on_task_change=args.init_cur_at_task_change,  reset_alpha_on_task_change=args.init_alpha_at_task_change
                                   )
    pass
save_interval = args.save_interval
eval_interval = args.eval_interval
save_dir = args.save_directory

test_sample_no = 10

test_lengths = change_varaiable

if args.env_type == "roboschool":
    if args.env == "Walker2DPyBulletEnv-v0" or args.env == "HopperPyBulletEnv-v0":
        if args.env_change_type == "spur_flux" or args.env_change_type == "sine_flux":
            test_lengths = change_varaiable_test
print(test_lengths)
#results
results = [[] for i in range(len(test_lengths))]

state = A.initalize()


print(change_varaiable_at)

experiment_no = args.experiment_no
inital_step_no = 0

print("experiment_no = " + str(experiment_no))
print(args.env)

if args.load_from_old:
    c = args.load_index
    save_dir_temp = save_dir + "/e" + str(experiment_no)


    if args.algo == "SAC_w_cur_buffer" or args.algo == "SAC_test":
        A.load(save_dir+"/q1", save_dir+"/q2",
                   save_dir_temp+"/q1_target", save_dir_temp+"/q2_target",
                   save_dir_temp+"/policy_target", icm_state_path=save_dir_temp+"/icm_state", icm_action_path=save_dir_temp+"/icm_action")
    elif args.algo == "DDPG":
        A.load( critic_path=save_dir+"/q", critic_target_path=save_dir_temp+"/q_target",
             policy_path=save_dir_temp+"/policy",
                policy_target_path=save_dir_temp+"/policy_target")

    elif args.algo == "DDPG_w_cur_buffer":
        A.load(critic_path=save_dir + "/q", critic_target_path=save_dir_temp + "/q_target",
               policy_path=save_dir_temp + "/policy",
               policy_target_path=save_dir_temp + "/policy_target",
               icm_state_path=save_dir_temp + "/icm_state", icm_action_path=save_dir_temp + "/icm_action",
               )
    else:
        A.load(save_dir_temp + "/q1", save_dir_temp + "/q2",
               save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
               save_dir_temp + "/policy_target")


    A.replay_buffer = torch.load(save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c))
    A.replay_buffer.tiebreaker = count(change_varaiable_at[c])

    results = torch.load( "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(
                    args.save_interval) + "_" + str(experiment_no))

    inital_step_no = change_varaiable_at[c]


ratio = 0.4
l_r = 20000


old_l = None
x = 0

for i in range(inital_step_no, args.no_steps):

    if i%1000==0:
        print(i)

    if args.env_type == "classic_control":
        if args.env_change_type == "sine_flux":
            new_l = schedule_generator.get_param(i)
            if args.env == "Cartpole-v0_masspole":
                A.env.set_mass_pole(mass=new_l)
            else:
                A.env.set_length(length=new_l)



        elif args.env_change_type == "spur_flux":
            new_l = schedule_generator.get_param(i)

            if args.env == "Cartpole-v0_masspole":
                A.env.set_mass_pole(mass=new_l)
            else:
                A.env.set_length(length=new_l)

            if new_l != old_l:
                old_l = new_l
                x += 1
                print("saving the replay memory")
                torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(x))

        elif args.env_change_type == "linear":

            new_l = schedule_generator.get_param(i)

            if args.env == "Cartpole-v0_masspole":
                A.env.set_mass_pole(mass=new_l)
            else:
                A.env.set_length(length=new_l)


        elif args.env_change_type == "discrete":
            if i == change_varaiable_at[c]:
                # save_the_buffer
                print("saving the replay memory")
                torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c))

                if args.env == "Cartpole-v0_masspole":
                    A.env.set_mass_pole(mass=change_varaiable[c])
                else:
                    A.env.set_length(length=change_varaiable[c])

                if c < len(change_varaiable_at) - 1:
                    c += 1

    else:

        if args.env_change_type == "sine_flux":
            new_power = schedule_generator.get_param(i)

            if args.env == "HopperPyBulletEnv-v0":
                A.env.change_power(new_power)


        elif args.env_change_type == "discrete" or args.env_change_type == "spur_flux":

            if i == change_varaiable_at[c]:
                # save_the_buffer
                print("saving the replay memory")
                torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c))

                if args.env_type == "roboschool":
                    A.env = env[c]

                if c < len(change_varaiable_at)-1:
                    c += 1




    if args.restart_alpha:
        if i%args.restart_alpha_interval == 0:
            A.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            A.alpha_optim = torch.optim.Adam([A.log_alpha], lr=A.alpha_lr)
    if args.algo == "SAC_test":
        r_i = ratio + (1 - ratio) * (l_r - min(i + 1000, l_r - 1000)) / l_r

        A.update(factor=r_i)
    else:
        A.update()



    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)


    if i%save_interval == 0:
        if i != 0:
            torch.save(results, "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(
                args.eval_interval) + "_" + str(experiment_no))
            save_dir_temp = save_dir + "/e" + str(experiment_no)

            if args.algo == "SAC_w_cur_buffer":
                A.save(save_dir_temp + "/q1", save_dir_temp + "/q2",
                       save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
                       save_dir_temp + "/policy_target", icm_state_path=save_dir_temp + "/icm_state",
                       icm_action_path=save_dir_temp + "/icm_action")
            elif args.algo == "Q_Learning" or args.algo == "Q_Learning_w_cur_buffer":
                A.save(save_dir_temp + "/q",
                       save_dir_temp + "/q_target")
            elif args.algo == "DDPG_w_cur_buffer":
                A.save(critic_path=save_dir + "/q", critic_target_path=save_dir_temp + "/q_target",
                       policy_path=save_dir_temp + "/policy",
                       policy_target_path=save_dir_temp + "/policy_target",
                       icm_state_path=save_dir_temp + "/icm_state", icm_action_path=save_dir_temp + "/icm_action",
                       )
            elif args.algo == "DDPG":
                A.save(critic_path=save_dir + "/q", critic_target_path=save_dir_temp + "/q_target",
                       policy_path=save_dir_temp + "/policy",
                       policy_target_path=save_dir_temp + "/policy_target")
            else:
                A.save(save_dir_temp + "/q1", save_dir_temp + "/q2",
                       save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
                       save_dir_temp + "/policy_target")


    if i%30000 == 0:
        if i != 0 or i==0:
            if args.algo == "SAC_w_cur" or args.algo == "SAC_w_cur_buffer" or args.algo == "SAC_test" or args.algo == "DDPG_w_cur_buffer" or args.algo == "Q_Learning_w_cur_buffer":
                torch.save(A.icm_i_r, "results/native_SAC_catastrophic_forgetting/inverse_curiosity" + str(experiment_no))
                torch.save(A.icm_f_r, "results/native_SAC_catastrophic_forgetting/forward_curiosity" + str(experiment_no))
                torch.save(A.icm_r, "results/native_SAC_catastrophic_forgetting/reward_curiosity" + str(experiment_no))
    if i % 30000 == 0:
        if i != 0:
            if args.algo == "SAC_w_cur_buffer":
                torch.save(A.alpha_history,
                           "results/native_SAC_catastrophic_forgetting/alpha/alpha_history" + str(experiment_no))

    if i%eval_interval==0:
        if args.env_type == "classic_control":
            if args.env == "Cartpole-v0_masspole":
                print("current variable = " + str(A.env.m_p))
            else:
                print("current variable = " + str(A.env.l))
        elif args.env_type == "roboschool":

            if args.env == "ReacherPyBulletEnv-v0":
                print("current variable = " + str(A.env.torque_factor))
            elif args.env == "InvertedPendulumSwingupPyBulletEnv-v0":
                print("current variable = " + str(A.env.length))
            elif args.env  == "Walker2DPyBulletEnv-v0_leg_len":
                print("current variable = " + str(A.env.l_length))
            else:
                print("current variable = " + str(A.env.power))



        for l_i, l in enumerate(test_lengths):
            rew_total = 0
            for k in range(test_sample_no):
                if args.env_type == "classic_control":
                    e = env_eval

                    if args.env_change_type == "discrete":
                        e.set_length(l)
                    elif args.env_change_type == "sine_flux":
                        e.set_length(l)
                    elif args.env_change_type == "spur_flux":
                        e.set_length(l)
                    elif args.env_change_type == "linear":
                        e.set_length(l)
                elif args.env_type == "roboschool":
                    e = env_eval[l_i]


                s = e.reset()
                rew = 0
                for j in range(A.max_episodes):

                    a = A.get_action(s, evaluate=True)
                    s, r, d, _ = e.step(a)
                    rew += r
                    #e.render()
                    if d == True:
                        break
                rew_total += rew

            rew_total = rew_total/test_sample_no
            results[l_i].append(rew_total)

            if args.algo == "SAC_w_cur" or args.algo == "SAC_w_r_cur":
                pass


            #print("reward at itr " + str(i) + " = " + str(rew_total) + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) + " for length: " + str(l))
            print("reward at itr " + str(i) + " = " + str(rew_total) +  " for variable: " + str(l))



    #saving replay buffer after time scales
    if args.save_buff_after == -1:
        pass
    else:
        if i%args.save_buff_after == 0:
            torch.save(A.replay_buffer, "buffer_saved_at_time_scale/replay_mem" + str(i))


#saving the final buffer
torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c+1))
torch.save(results, "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(args.eval_interval) + "_" + str(experiment_no))

if args.algo == "SAC_w_cur" or args.algo == "SAC_w_cur_buffer" or args.algo == "SAC_test" or args.algo == "DDPG_w_cur_buffer" or args.algo == "Q_Learning_w_cur_buffer":
    torch.save(A.icm_i_r, "results/native_SAC_catastrophic_forgetting/inverse_curiosity" + str(experiment_no))
    torch.save(A.icm_f_r, "results/native_SAC_catastrophic_forgetting/forward_curiosity" + str(experiment_no))
    torch.save(A.icm_r, "results/native_SAC_catastrophic_forgetting/reward_curiosity" + str(experiment_no))


if args.algo == "SAC_w_cur_buffer":
    torch.save(A.alpha_history, "results/native_SAC_catastrophic_forgetting/alpha/alpha_history" + str(experiment_no))