from custom_envs.pybulletgym_custom.envs.roboschool.envs.env_bases import BaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.manipulators.pusher import Pusher
from custom_envs.pybulletgym_custom.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np


class PusherBulletEnv(BaseBulletEnv):
    def __init__(self, gravity=9.81):
        self.robot = Pusher()
        BaseBulletEnv.__init__(self, self.robot)

        self.gravity = 9.81

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=self.gravity, timestep=0.0020, frame_skip=5)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        joint_vel = np.array([
            self.robot.shoulder_pan_joint.get_velocity(),
            self.robot.shoulder_lift_joint.get_velocity(),
            self.robot.upper_arm_roll_joint.get_velocity(),
            self.robot.elbow_flex_joint.get_velocity(),
            self.robot.forearm_roll_joint.get_velocity(),
            self.robot.wrist_flex_joint.get_velocity(),
            self.robot.wrist_roll_joint.get_velocity()
        ])

        action_product = np.matmul(np.abs(a), np.abs(joint_vel))
        action_sum = np.sum(a)

        electricity_cost = (
                -0.10 * action_product  # work torque*angular_velocity
                - 0.01 * action_sum  # stall torque require some energy
        )

        stuck_joint_cost = 0
        for j in self.robot.ordered_joints:
            if np.abs(j.current_relative_position()[0]) - 1 < 0.01:
                stuck_joint_cost += -0.1

        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.HUD(state, a, False)
        return state, sum(self.rewards), False, {}

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
