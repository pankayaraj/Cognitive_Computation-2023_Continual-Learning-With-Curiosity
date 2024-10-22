from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from custom_envs.pybulletgym_custom.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import numpy as np


class HalfCheetah(WalkerBase, MJCFBasedRobot):
    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self, power, delta=0.0 ):
        WalkerBase.__init__(self, power=power)
        MJCFBasedRobot.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=26)
        self.delta = delta
    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.jdict["bthigh"].power_coef = 120.0 + self.delta
        self.jdict["bshin"].power_coef  = 90.0 #+ self.delta
        self.jdict["bfoot"].power_coef  = 60.0 #+ self.delta
        self.jdict["fthigh"].power_coef = 140.0 + self.delta
        self.jdict["fshin"].power_coef  = 60.0 #+ self.delta
        self.jdict["ffoot"].power_coef  = 30.0 #+ self.delta
