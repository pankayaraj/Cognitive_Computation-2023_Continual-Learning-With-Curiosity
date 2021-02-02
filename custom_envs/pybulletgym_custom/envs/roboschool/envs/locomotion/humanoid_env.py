from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import Humanoid


class HumanoidBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, robot=Humanoid(), power=0.41):
        #self.robot = robot
        self.power = power

        self.robot = Humanoid(power=power)
        WalkerBaseBulletEnv.__init__(self, self.robot)
        self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost

