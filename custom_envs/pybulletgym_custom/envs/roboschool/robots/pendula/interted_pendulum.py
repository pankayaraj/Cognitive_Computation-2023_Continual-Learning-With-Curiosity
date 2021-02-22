from custom_envs.pybulletgym_custom.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import numpy as np
import xml.etree.ElementTree
import os
class InvertedPendulum(MJCFBasedRobot):
    swingup = False

    def __init__(self, torque_factor=100, friction="1 0.1 0.1", length = 0.6, index=0):

        self.friction = friction

        self.length = length
        self.from_to = "0 0 0 0.001 0 " + str(length)


        self.model_xml = 'inverted_pendulum.xml'
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", self.model_xml)
        tree = xml.etree.ElementTree.parse(full_path)
        root = tree.getroot()

        root[4][1][2][1].set("fromto", self.from_to)

        new_xml = "custom_inverted_pendulum/inverted_pendulum" + str(index) + ".xml"
        new_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mjcf", new_xml)

        root[1][1].set("friction", self.friction)
        tree.write(new_path)

        #MJCFBasedRobot.__init__(self, 'inverted_pendulum.xml', 'cart', action_dim=1, obs_dim=5)
        MJCFBasedRobot.__init__(self, new_xml, 'cart', action_dim=1, obs_dim=5)

        self.torque_factor=torque_factor

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position( u if not self.swingup else 3.1415+u , 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0

        self.slider.set_motor_torque(  self.torque_factor*float(np.clip(a[0], -1, +1)) )

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        x, vx = self.slider.current_position()
        assert( np.isfinite(x) )

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([
            x, vx,
            np.cos(self.theta), np.sin(self.theta), theta_dot
        ])


class InvertedPendulumSwingup(InvertedPendulum):
    swingup = True
    def __init__(self,  torque_factor=100, friction="1 0.1 0.1", length = 0.6, index=0):
        InvertedPendulum.__init__(self, torque_factor=torque_factor, friction=friction, length=length, index=index)

