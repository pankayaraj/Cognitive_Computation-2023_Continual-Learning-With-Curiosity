

from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import Hopper
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors.walker_base import WalkerBase


class HopperBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, power = 0.75):
        self.power = power
        self.robot = Hopper(power=power)
        WalkerBaseBulletEnv.__init__(self, self.robot)
	


