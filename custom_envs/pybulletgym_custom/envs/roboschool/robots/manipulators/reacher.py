from custom_envs.pybulletgym_custom.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import numpy as np


class Reacher(MJCFBasedRobot):
    TARG_LIMIT_MAX = 0.27
    TARG_LIMIT_MIN = -0.27


    def __init__(self, torque_factor= 0.05):
        MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)
        self.torque_factor = torque_factor
    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=self.TARG_LIMIT_MIN, high=self.TARG_LIMIT_MAX), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=self.TARG_LIMIT_MIN, high=self.TARG_LIMIT_MAX), 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint1"]
        self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(self.torque_factor * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(self.torque_factor* float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)
