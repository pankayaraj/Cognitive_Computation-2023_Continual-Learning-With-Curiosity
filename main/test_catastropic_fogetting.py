import torch
import numpy as np

import argparse

from algorithms.SAC import SAC
from algorithms.SAC_w_Curiosity import SAC_with_Curiosity

from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

from custom_envs.custom_pendulum import PendulumEnv


parser = argparse.ArgumentParser(description='SAC arguments')

parser.add_argument("--algo", type=str, default="SAC_w_cur")
parser.add_argument("--env", type=str, default="Pendulum-v0")
parser.add_argument("--policy", type=str, default="gaussian")
parser.add_argument("--hidden_layers", type=list, default=[256, 256])
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--automatic_entropy_tuning", type=bool, default=True)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--memory_size", type=int, default=90000)
parser.add_argument("--no_steps", type=int, default=90000)
parser.add_argument("--max_episodes", type=int, default=200)
parser.add_argument("--save_directory", type=str, default="models/native_SAC_catastropic_forgetting/diff_length")

parser.add_argument("--interval_based_increment", type=bool, default=True,
                    help="weather to increase the factor on certain intervals or do it linearly")

parser.add_argument("--rate_change_interval", type=int, default=30000)
parser.add_argument("--l_interval_rate", type=int, default=0.2,
                    help="rate to increase length by for every rate change interval")
parser.add_argument("--l_linear_rate", type=float, default=65e-7,
                    help="rate of change for linear increase")


args = parser.parse_args()

env, env_eval = None, None
if args.env == "Pendulum-v0":
    env = PendulumEnv()
    env_eval = PendulumEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

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

if args.algo == "SAC":
    A = SAC(env, q_nn_param, policy_nn_param, algo_nn_param,
        max_episodes=args.max_episodes, memory_capacity=args.memory_size,
        batch_size=args.batch_size, alpha_lr=args.lr)
elif args.algo == "SAC_w_cur":
    A = SAC_with_Curiosity(env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=args.max_episodes,
                           memory_capacity=args.memory_size
                           , batch_size=args.batch_size, alpha_lr=args.lr)


save_interval = args.save_interval
eval_interval = args.eval_interval
save_dir = args.save_directory

test_sample_no = 10


test_lengths = [1.0 for i in  range(0, int(args.no_steps/args.rate_change_interval))]
if args.interval_based_increment:
    for i in range(1, int(args.no_steps / args.rate_change_interval)):
        test_lengths[i] = float(test_lengths[i-1] + args.l_interval_rate)
else:
    for i in range(1, int(args.no_steps / args.rate_change_interval)):
        test_lengths[i] = test_lengths[i-1] + args.l_linear_rate*args.rate_change_interval

#results
results = [[] for i in range(len(test_lengths))]


state = A.initalize()

for i in range(args.no_steps):



    if args.interval_based_increment:
        if i%args.rate_change_interval == 0:
            print("Length Change")
            if i != 0:
                A.env.set_length(length=env.l + args.l_interval_rate)
            #A.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            #A.alpha_optim = torch.optim.Adam([A.log_alpha], lr=A.alpha_lr)
    else:
        if i != 0:
            A.env.set_length(length=env.l + args.l_linear_rate)

    if i%args.memory_size:
        #A.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        #A.alpha_optim = torch.optim.Adam([A.log_alpha], lr=A.alpha_lr)
        pass
    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        A.save(save_dir+"/q1", save_dir+"/q2",
               save_dir+"/q1_target", save_dir+"/q2_target",
               save_dir+"/policy_target")


    if i%eval_interval==0:
        A.debug.print_all()

        for l_i, l in enumerate(test_lengths):
            rew_total = 0
            for k in range(test_sample_no):
                e = env_eval
                #e = PendulumEnv()
                e.set_length(l)

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


            print("reward at itr " + str(i) + " = " + str(rew_total) + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) + " for length: " + str(l))

torch.save(A.replay_buffer, save_dir + "/replay_mem")
torch.save(results, "results/native_SAC_catastrophic_forgetting/results_length__s_i_" + str(args.save_interval) + "_3")
