from custom_envs.pybulletgym_custom.envs.roboschool.envs.env_bases import BaseBulletEnv
from custom_envs.pybulletgym_custom.envs.roboschool.robots.pendula.interted_pendulum import InvertedPendulum, InvertedPendulumSwingup
from custom_envs.pybulletgym_custom.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np


class InvertedPendulumBulletEnv(BaseBulletEnv):
    def __init__(self, torque_factor=100, gravity=9.8, friction = "1 0.1 0.1", length= 0.6, index=0):

        self.torque_factor = torque_factor
        self.gravity = gravity
        self.friction = friction
        self.length =  length
        self.index = index

        self.robot = InvertedPendulum(torque_factor=torque_factor, friction=self.friction, length=self.length,
                                      index=self.index)
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=self.gravity, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        vel_penalty = 0
        if self.robot.swingup:
            reward = np.cos(self.robot.theta)
            done = False
        else:
            reward = 1.0
            done = np.abs(self.robot.theta) > .2
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)


class InvertedPendulumSwingupBulletEnv(InvertedPendulumBulletEnv):
    def __init__(self,  torque_factor=100, gravity=9.8, friction = "1 0.1 0.1", length= 0.6, index=0):

        self.gravity = gravity
        self.length = length
        self.robot = InvertedPendulumSwingup(torque_factor=torque_factor, friction=friction, length=length, index=index )
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
