from custom_envs.pybulletgym_custom.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors import Atlas
from custom_envs.pybulletgym_custom.envs.roboschool.scenes import StadiumScene


class AtlasBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, power = 2.9):
        self.power = power
        self.robot = Atlas(power=power)
        WalkerBaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165/8, frame_skip=8)   # 8 instead of 4 here
        return self.stadium_scene

    def robot_specific_reset(self):
        self.robot.robot_specific_reset()
