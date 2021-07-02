from custom_envs.pybulletgym_custom.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from custom_envs.pybulletgym_custom.envs.roboschool.robots.robot_bases import MJCFBasedRobot

import os
import xml.etree.ElementTree

class Hopper(WalkerBase, MJCFBasedRobot):
    foot_list = ["foot"]

    def __init__(self, power=0.75, leg_length = 0.5, thigh_length= 0.45, foot_length = 0.40, leg_size = 0.04,
                 thigh_size=0.05, index=0):
        self.power = power

        self.leg_length = leg_length
        self.thigh_length = thigh_length
        self.foot_length = foot_length

        self.leg_size = str(leg_size)
        self.thigh_size = str(thigh_size)

        self.from_troso = "0 0 " + str(thigh_length + 0.1 + leg_length + 0.4)
        self.to_troso = "0 0 " + str(thigh_length + 0.1 + leg_length)
        self.from_to_troso = self.from_troso + " " + self.to_troso

        self.from_t = "0 0 " + str(thigh_length + 0.1 + leg_length)
        self.to_t =   "0 0 " + str(0.1 + leg_length)
        self.from_to_t = self.from_t + " " + self.to_t

        self.from_l = "0 0 "+ str(0.1 + leg_length)
        self.to_l = "0 0 0.1"
        self.from_to_l =  self.from_l + " " + self.to_l

        self.from_f = str(-foot_length/2)  + " 0 0.1"
        self.to_f = str(foot_length/2) +  " 0 0.1"
        self.from_to_f = self.from_f + " " +self.to_f



        self.model_xml = "hopper.xml"
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", self.model_xml)
        tree = xml.etree.ElementTree.parse(full_path)
        root = tree.getroot()

        root[3][0][3].set("fromto", self.from_to_troso)

        root[3][0][4][1].set("fromto", self.from_to_t)
        root[3][0][4][0].set("pos", self.from_t)
        root[3][0][4][1].set("size", self.thigh_size)

        root[3][0][4][2][1].set("fromto", self.from_to_l)
        root[3][0][4][2][0].set("pos", self.from_l)
        root[3][0][4][2][1].set("size", self.leg_size)


        root[3][0][4][2][2][1].set("fromto", self.from_to_f)
        root[3][0][4][2][2][1].set("pos", self.to_l)

        new_xml = "custom_hopper/hopper" + str(index) + ".xml"

        new_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", new_xml)

        tree.write(new_path)

        self.walker_base = WalkerBase
        self.walker_base.__init__(self, power=power)
        MJCFBasedRobot.__init__(self, new_xml, "torso", action_dim=3, obs_dim=15)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
