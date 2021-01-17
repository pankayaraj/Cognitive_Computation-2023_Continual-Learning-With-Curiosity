import torch
import numpy as np
from itertools import count

import argparse

from algorithms.SAC import SAC
from algorithms.SAC_w_Curiosity import SAC_with_Curiosity
from algorithms.SAC_w_Reward_based_Curiosity import SAC_with_reward_based_Curiosity
from algorithms.SAC_w_Cur_Buffer import SAC_with_Curiosity_Buffer

from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

from custom_envs.custom_pendulum import PendulumEnv
from util.roboschool_util.make_new_env import make_array_env

parser = argparse.ArgumentParser(description='SAC arguments')
#"SAC_w_cur_buffer"
#Half_Reservior_FIFO
#Half_Reservior_TR_FIFO_Flow_Through
#"Half_Reservior_FIFO_with_FT"
#"HopperPyBulletEnv-v0"
#"Walker2DPyBulletEnv-v0"

parser.add_argument("--algo", type=str, default="SAC_w_cur_buffer")
parser.add_argument("--buffer_type", type=str, default="Half_Reservior_FIFO_with_FT")
parser.add_argument("--env", type=str, default="HopperPyBulletEnv-v0")
parser.add_argument("--env_type", type=str, default="roboschool")

parser.add_argument("--load_from_old", type=bool, default=False)
parser.add_argument("--load_index", type=int, default=3) #to indicate which change of varaiable we are at
parser.add_argument("--starting_time_step", type=int, default=0) #from which time fram to start things

parser.add_argument("--experiment_no", type=int, default=6)

#parser.add_argument("--algo", type=str, default="SAC_w_cur_buffer")
#parser.add_argument("--buffer_type", type=str, default="Half_Reservior_FIFO_with_FT")
#parser.add_argument("--env", type=str, default="Pendulum-v0")
#parser.add_argument("--env_type", type=str, default="classic_control")

#parser.add_argument("--fifo_frac", type=float, default=0.34)
parser.add_argument("--fifo_frac", type=float, default=0.05)
parser.add_argument("--no_curiosity_networks", type=int, default=3)

parser.add_argument("--policy", type=str, default="gaussian")
parser.add_argument("--hidden_layers", type=list, default=[256, 256])
parser.add_argument("--lr", type=float, default=0.0003)
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
parser.add_argument("--memory_size", type=int, default=50000)
#parser.add_argument("--memory_size", type=int, default=8000)
#parser.add_argument("--no_steps", type=int, default=250000)
parser.add_argument("--no_steps", type=int, default=400000)
parser.add_argument("--max_episodes", type=int, default=1000)
#parser.add_argument("--max_episodes", type=int, default=200)
parser.add_argument("--save_directory", type=str, default="models/native_SAC_catastropic_forgetting/diff_length")

#hopper
#change_varaiable_at = [1, 100000, 150000, 350000] #v1
#change_varaiable = [0.75, 1.75, 2.75, 3.75]

#change_varaiable_at = [1, 50000, 250000, 300000, 350000] #v2
#change_varaiable = [0.75, 2.75, 4.75,  6.75, 8.75]

change_varaiable_at = [1, 50000, 350000] #v3
#change_varaiable_at = [1, 100000, 500000, 600000, 700000]
#change_varaiable = [0.75, 4.75, 8.75,  2.75, 6.75]
change_varaiable = [0.75, 4.75, 8.75]
#pendulum
#change_varaiable_at = [1, 30000, 60000, 120000, 200000]
#change_varaiable = [1.0, 1.2, 1.4, 1.6, 1.8]

#walker2D
#change_varaiable_at = [1, 100000, 150000, 350000, 400000]
#change_varaiable = [0.40, 1.40, 2.40, 3.40, 4.40]


c = 0

args = parser.parse_args()

if args.env_type == "classic_control":
    if args.env == "Pendulum-v0":
        env = PendulumEnv()
        env_eval = PendulumEnv()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        ini_env = env

elif args.env_type == "roboschool":
    env, env_eval = make_array_env(change_varaiable, args.env)
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
algo_nn_param = Algo_Param(gamma=args.gamma, alpha=args.alpha, tau=args.tau,
                           target_update_interval=args.target_update_interval,
                           automatic_alpha_tuning=args.automatic_entropy_tuning)
buffer_type = args.buffer_type

if args.algo == "SAC":
    A = SAC(ini_env, q_nn_param, policy_nn_param, algo_nn_param,
        max_episodes=args.max_episodes, memory_capacity=args.memory_size,
        batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type, fifo_frac=args.fifo_frac)
elif args.algo == "SAC_w_cur":
    A = SAC_with_Curiosity(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=args.max_episodes,
                           memory_capacity=args.memory_size
                           , batch_size=args.batch_size, alpha_lr=args.lr)
elif args.algo == "SAC_w_r_cur":
    A = SAC_with_reward_based_Curiosity(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=args.max_episodes,
                           memory_capacity=args.memory_size
                           , batch_size=args.batch_size, alpha_lr=args.lr)
elif args.algo == "SAC_w_cur_buffer":
    A = SAC_with_Curiosity_Buffer(ini_env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param,
                           max_episodes=args.max_episodes,
                           memory_capacity=args.memory_size
                           , batch_size=args.batch_size, alpha_lr=args.lr, buffer_type=buffer_type, fifo_frac=args.fifo_frac
                                  , no_cur_network=args.no_curiosity_networks)

save_interval = args.save_interval
eval_interval = args.eval_interval
save_dir = args.save_directory

test_sample_no = 10
test_lengths = change_varaiable

#results
results = [[] for i in range(len(test_lengths))]

state = A.initalize()



experiment_no = args.experiment_no
inital_step_no = 0

print("experiment_no = " + str(experiment_no))

if args.load_from_old:
    c = args.load_index
    save_dir_temp = save_dir + "/e" + str(experiment_no)


    if args.algo == "SAC_w_cur_buffer":
        A.load(save_dir+"/q1", save_dir+"/q2",
                   save_dir_temp+"/q1_target", save_dir_temp+"/q2_target",
                   save_dir_temp+"/policy_target", icm_state_path=save_dir_temp+"/icm_state", icm_action_path=save_dir_temp+"/icm_action")
    else:
        A.load(save_dir_temp + "/q1", save_dir_temp + "/q2",
               save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
               save_dir_temp + "/policy_target")


    A.replay_buffer = torch.load(save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c))
    A.replay_buffer.tiebreaker = count(change_varaiable_at[c])

    results = torch.load( "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(
                    args.save_interval) + "_" + str(experiment_no))

    inital_step_no = change_varaiable_at[c]

for i in range(inital_step_no, args.no_steps):

    if i%1000==0:
        print(i)

    if i%change_varaiable_at[c] == 0:
        # save_the_buffer
        torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c))

        if args.env_type == "classic_control":
            A.env.set_length(length=change_varaiable[c])
        elif args.env_type == "roboschool":
            A.env = env[c]

        if c < len(change_varaiable_at)-1:
            c += 1



    if args.restart_alpha:
        if i%args.restart_alpha_interval == 0:
            A.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            A.alpha_optim = torch.optim.Adam([A.log_alpha], lr=A.alpha_lr)

    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)


    if i%save_interval == 0:

        torch.save(results, "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(
            args.eval_interval) + "_" + str(experiment_no))
        save_dir_temp = save_dir + "/e" + str(experiment_no)

        if args.algo == "SAC_w_cur_buffer":
            A.save(save_dir_temp + "/q1", save_dir_temp + "/q2",
                   save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
                   save_dir_temp + "/policy_target", icm_state_path=save_dir_temp + "/icm_state",
                   icm_action_path=save_dir_temp + "/icm_action")
        else:
            A.save(save_dir_temp + "/q1", save_dir_temp + "/q2",
                   save_dir_temp + "/q1_target", save_dir_temp + "/q2_target",
                   save_dir_temp + "/policy_target")



    if i%eval_interval==0:
        if args.env_type == "classic_control":
            print("current variable = " + str(A.env.l))
        elif args.env_type == "roboschool":
            print("current variable = " + str(A.env.power))


        for l_i, l in enumerate(test_lengths):
            rew_total = 0
            for k in range(test_sample_no):
                if args.env_type == "classic_control":
                    e = env_eval
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

#saving the final buffer
torch.save(A.replay_buffer, save_dir + "/e" + str(experiment_no) + "/replay_mem" + str(c+1))
torch.save(results, "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(args.eval_interval) + "_" + str(experiment_no))

if args.algo == "SAC_w_cur" or args.algo == "SAC_w_cur_buffer":
    torch.save(A.icm_i_r, "results/native_SAC_catastrophic_forgetting/inverse_curiosity" + str(experiment_no))
    torch.save(A.icm_f_r, "results/native_SAC_catastrophic_forgetting/forward_curiosity" + str(experiment_no))
    #torch.save(A.icm_r_r, "results/native_SAC_catastrophic_forgetting/reward_curiosity" + str(experiment_no))