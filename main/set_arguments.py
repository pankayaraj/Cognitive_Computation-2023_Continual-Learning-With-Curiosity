


def set_arguments(args):


    args.do_irm = False


    if args.supersede_env == "Pendulum":
        args.env_type = "classic_control"
        args.env = "Pendulum-v0"

        args.batch_size = 512
        args.memory_size = 20000
        #args.no_steps = 150000
        args.no_steps = 300000


        args.max_episodes = 200
        args.hidden_layers = [256, 256]

        args.no_curiosity_networks = 1
        args.fow_cur_weight = 0.0
        args.inv_cur_weight = 1.0
        args.rew_cur_weight = 0.0
        args.n_k = 600
        args.l_k = 8000
        args.m_f = 1.5

        args.min_lim = 1.0
        args.max_lim = 1.8
        args.factor = 0.0001
        args.sche_steps = 400000


        args = set_algo(args)


        #change_varaiable_at = [1, 20000, 120000]
        #change_varaiable = [1.0, 1.4, 1.8]

        change_varaiable_at = [1, 20000, 90000, 160000, 230000]
        change_varaiable = [1.0, 1.4, 1.5, 1.55, 1.8]



    elif args.supersede_env == "Cartpole":
        args.env_type = "classic_control"
        args.env = "Cartpole-v0"

        args.batch_size = 32
        args.memory_size = 20000
        args.no_steps = 180000
        args.max_episodes = 200
        args.hidden_layers = [64, 64]

        args.no_curiosity_networks = 1
        args.fow_cur_weight = 0.0
        args.inv_cur_weight = 1.0
        args.rew_cur_weight = 0.0
        args.n_k = 600
        args.l_k = 8000
        args.m_f = 10.0

        args = set_algo(args)

        change_varaiable_at = [1, 20000]
        change_varaiable = [0.5, 10.5]


    elif args.supersede_env == "Hopper":
        args.env_type = "roboschool"
        args.env = "HopperPyBulletEnv-v0"

        args.batch_size = 512
        args.memory_size = 50000
        args.no_steps = 400000
        args.max_episodes = 1000
        args.hidden_layers = [256, 256]

        args.no_curiosity_networks = 3
        #args.no_curiosity_networks = 1

        args.fow_cur_weight = 0.0
        args.inv_cur_weight = 1.0
        args.inv_cur_weight = 0.0
        #args.rew_cur_weight = 0.0
        #args.rew_cur_weight = 1.0

        args.n_k = 500
        args.l_k = 30000
        args.m_f = 2.5

        args.min_lim = 0.75
        args.max_lim = 8.75
        args.factor = 0.00003
        args.sche_steps = 400000

        args = set_algo(args)

        change_varaiable_at = [1, 50000, 350000]
        change_varaiable = [0.75, 4.75, 8.75]

    elif args.supersede_env == "Walker2D":
        args.env_type = "roboschool"
        args.env = "Walker2DPyBulletEnv-v0"

        args.batch_size = 512
        args.memory_size =100000
        args.no_steps = 400000
        args.max_episodes = 1000
        args.hidden_layers = [256, 256]

        args.no_curiosity_networks = 1
        args.fow_cur_weight = 0.0
        args.inv_cur_weight = 1.0
        args.rew_cur_weight = 0.05
        args.n_k = 2000
        args.l_k = 30000
        args.m_f = 0.2

        args = set_algo(args)

        change_varaiable_at = [1, 250000, 350000]
        change_varaiable = [1.40, 7.40, 13.40, ]

    return args, change_varaiable_at, change_varaiable


def set_algo(args):


    if args.supersede_buff == "FIFO":
        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning"
        else:
            args.algo = "SAC"


        args.buffer_type = "FIFO"
    elif args.supersede_buff == "HRF":

        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning"
        else:
            args.algo = "SAC"

        args.buffer_type = "Half_Reservior_FIFO_with_FT"
        args.priority = "uniform"




    elif args.supersede_buff == "MTR_low":
        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning"
            args.mtr_buff_no = 2
        else:
            args.algo = "SAC"
            args.mtr_buff_no = 3

        args.buffer_type = "MTR"

    elif args.supersede_buff == "MTR_high":
        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning"

        else:
            args.algo = "SAC"


        args.buffer_type = "MTR"
        args.mtr_buff_no = 5

    elif args.supersede_buff == "MTR_high_20":
        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning"

        else:
            args.algo = "SAC"

        args.buffer_type = "MTR"
        args.mtr_buff_no = 20

    elif args.supersede_buff == "TS_HRF":

        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning_w_cur_buffer"
        else:
            args.algo = "SAC_w_cur_buffer"

        args.buffer_type = "Half_Reservior_FIFO_with_FT"
        args.priority = "uniform"

    elif args.supersede_buff == "HCRRF":

        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning_w_cur_buffer"
        else:
            args.algo = "SAC_w_cur_buffer"

        args.buffer_type = "Hybrid_Cur_Res_Res"
        args.priority = "uniform"

    elif args.supersede_buff == "TS_C_HRF":

        if args.env == "Cartpole-v0":
            args.algo = "Q_Learning_w_cur_buffer"
        else:
            args.algo = "SAC_w_cur_buffer"



        args.buffer_type = "Half_Reservior_FIFO_with_FT"
        args.priority = "curiosity"
    return args