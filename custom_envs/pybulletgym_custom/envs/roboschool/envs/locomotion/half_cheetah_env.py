from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import HalfCheetah


class HalfCheetahBulletEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = HalfCheetah()
        WalkerBaseBulletEnv.__init__(self, self.robot)
