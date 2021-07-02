from gym.envs.registration import register
import custom_envs.pybulletgym_custom
import gym

def make_array_env(change_variable, name, change_env_type, test_var = None):
    env, env_eval = [], []
    print("enviornment = " + name)
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
            if change_env_type != "spur_flux" and change_env_type != "sine_flux":
                env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i+1)))
                env_eval[i].reset()

            env[i].reset()


        elif name == "HopperPyBulletEnv-v0_leg":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"leg_length": change_variable[i], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HopperPyBulletEnv-v0_leg_size":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"leg_size": change_variable[i], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HopperPyBulletEnv-v0_leg_size_length":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"leg_size": change_variable[i][0], "leg_length": change_variable[i][1], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()



        elif name == "HopperPyBulletEnv-v0_thigh":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"thigh_length": change_variable[i], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HopperPyBulletEnv-v0_thigh_size":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"thigh_size": change_variable[i], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HopperPyBulletEnv-v0_foot":

            register(id='HopperPyBulletEnv-v' + str(i+1),
                     entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                     kwargs={"foot_length": change_variable[i], "index": i},
                     max_episode_steps=1000,
                     reward_threshold=2500.0)

            env.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HopperPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()



        elif name == "Walker2DPyBulletEnv-v0":

            register(
                id='Walker2DPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('Walker2DPyBulletEnv-v' + str(i+1)))
            env_eval.append(gym.make('Walker2DPyBulletEnv-v' + str(i+1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "Walker2DPyBulletEnv-v0_leg_len":

            register(
                id='Walker2DPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker2d_env:Walker2DBulletEnv',
                kwargs={'leg_length': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('Walker2DPyBulletEnv-v' + str(i+1)))
            env_eval.append(gym.make('Walker2DPyBulletEnv-v' + str(i+1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "AntPyBulletEnv-v0":
            register(
                id='AntPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.ant_env:AntBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('AntPyBulletEnv-v' + str(i+1)))
            env_eval.append(gym.make('AntPyBulletEnv-v' + str(i+1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == 'AtlasPyBulletEnv-v0':
            register(
                id='AtlasPyBulletEnv-v' + str(i + 1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.atlas_env:AtlasBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('AtlasPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('AtlasPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HumanoidPyBulletEnv-v0":
            register(
                id="HumanoidPyBulletEnv-v" + str(i + 1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.humanoid_env:HumanoidBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=2500.0
            )

            env.append(gym.make('HumanoidPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HumanoidPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "HalfCheetahPyBulletEnv-v0":
            register(
                id="HalfCheetahPyBulletEnv-v" + str(i + 1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.half_cheetah_env:HalfCheetahBulletEnv',
                kwargs={'power': change_variable[i]},
                max_episode_steps=1000,
                reward_threshold=3000.0
            )

            env.append(gym.make('HalfCheetahPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('HalfCheetahPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "ReacherPyBulletEnv-v0":

            register(
                id='ReacherPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
                kwargs={"torque_factor":  change_variable[i]},
                max_episode_steps=150,
                reward_threshold=18.0,
            )

            env.append(gym.make('ReacherPyBulletEnv-v' + str(i+1)))
            env_eval.append(gym.make('ReacherPyBulletEnv-v' + str(i+1)))

            env[i].reset()
            env_eval[i].reset()

        elif name == "InvertedPendulumSwingupPyBulletEnv-v0":

            register(
                id='InvertedPendulumSwingupPyBulletEnv-v' + str(i+1),
                entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.pendulum.inverted_pendulum_env:InvertedPendulumSwingupBulletEnv',
                max_episode_steps=1000,
                kwargs={"length":  change_variable[i], "index": i},
                reward_threshold=950.0,
            )

            env.append(gym.make('InvertedPendulumSwingupPyBulletEnv-v' + str(i + 1)))
            env_eval.append(gym.make('InvertedPendulumSwingupPyBulletEnv-v' + str(i + 1)))

            env[i].reset()
            env_eval[i].reset()


    #for spur flux where multiple env repeat again and again
    if change_env_type == "spur_flux" or change_env_type == "sine_flux":

        for i in range(len(test_var)):

            if name == "HopperPyBulletEnv-v0":
                register(
                    id='HopperPyBulletEnv-v' + str(len(change_variable) + i + 1),
                    entry_point='custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.hopper_env:HopperBulletEnv',
                    kwargs={'power': test_var[i]},
                    max_episode_steps=1000,
                    reward_threshold=2500.0
                )


                env_eval.append(gym.make('HopperPyBulletEnv-v' + str(len(change_variable) + i + 1)))
                env_eval[i].reset()




    return  env, env_eval



