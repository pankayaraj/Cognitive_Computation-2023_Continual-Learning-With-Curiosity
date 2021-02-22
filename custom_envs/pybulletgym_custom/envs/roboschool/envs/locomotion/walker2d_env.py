from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import Walker2D


class Walker2DBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, power= 0.40, length= 0.1, index=0):

        self.length = length
        self.power = power

        self.robot = Walker2D(power=power, length=self.length,index=index)
        WalkerBaseBulletEnv.__init__(self, self.robot)

