from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import Ant


class AntBulletEnv(WalkerBaseBulletEnv ):
    def __init__(self, power = 2.5):
        self.power=power
        self.robot = Ant(power=power)
        WalkerBaseBulletEnv.__init__(self, self.robot)
