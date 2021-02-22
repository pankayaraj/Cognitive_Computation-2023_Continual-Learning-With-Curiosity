from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from custom_envs.pybulletgym_custom.envs.roboschool.robots.robot_bases import MJCFBasedRobot

import os
import numpy as np
import xml.etree.ElementTree


class Walker2D(WalkerBase, MJCFBasedRobot):
    foot_list = ["foot", "foot_left"]

    def __init__(self, power=0.40, length=0.1, index=0):
        self.length = length
        self.from_to = "-0.0 0 0.1 0.2 0 " + str(length)


        self.model_xml =  "walker2d.xml"
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", self.model_xml)
        tree = xml.etree.ElementTree.parse(full_path)
        root = tree.getroot()

        root[3][0][4][2][2][1].set("fromto", self.from_to)

        new_xml = "custom_walker2D/walker2d" + str(index) + ".xml"
        new_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", new_xml)

        tree.write(new_path)


        WalkerBase.__init__(self, power=power)
        MJCFBasedRobot.__init__(self, new_xml, "torso", action_dim=6, obs_dim=22)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0

