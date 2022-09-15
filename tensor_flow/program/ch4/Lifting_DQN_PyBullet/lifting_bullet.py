import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
import pybullet_data
import pybullet_utils.bullet_client as bc

#シミュレータクラスの設定
class LiftingBulletEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
  def __init__(self, renders=True, discrete_actions=True):
    self._renders = renders
    self._physics_client_id = -1
    self.gravity = 9.8
    self.x_threshold = 2.4#ラケットの移動範囲
    self.force_mag = 10#ラケットを移動させるための力
    high = np.array([self.x_threshold,np.finfo(np.float32).max, self.x_threshold, self.x_threshold, np.finfo(np.float32).max])
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(-high, high, dtype=np.float32)
    self.ballInitPosition = [1, 0, 1]#ボールの初期位置
    self.ballInitVelocity = [-2, 0, 0]#ボールの初期速度
    self.seed()
    self.reset()
    self.viewer = None
    self._configure()
#行動による状態変化
  def step(self, action):
    p = self._p
    force = self.force_mag if action==1 else -self.force_mag

    p.setJointMotorControl2(self.racket, 0, p.TORQUE_CONTROL, force=force)
    p.stepSimulation()

    ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
    ball_x, ball_y, ball_z = ball_pos
    ball_vel, _ = p.getBaseVelocity(self.ball)
    ball_vx, ball_vy, ball_vz, = ball_vel
    racket_x, racket_vx = p.getJointState(self.racket, 0)[0:2]

    reward = 0
    done = False
    self.state = [racket_x, racket_vx, ball_x, ball_z, ball_vx]
    done = False
    if ball_z < 0 or racket_x > self.x_threshold or racket_x < -self.x_threshold: 
      reward = -1.0
      done = True
    elif len(p.getContactPoints(self.racket, self.ball, 0, -1)) > 0:#ラケット(0)とボール(-1)の衝突検知
      reward = 1.0

    return np.array(self.state), reward, done, {}
#初期化
  def reset(self):
    if self._physics_client_id < 0:
      self._p = bc.BulletClient(connection_mode=p2.GUI)
      self._physics_client_id = self._p._client

      p = self._p
      p.resetSimulation()

      self.racket = p.loadURDF("racket.urdf", [0, 0, 0])   
      self.ball = p.loadURDF("ball.urdf", [0, 0, 0])

      p.changeDynamics(self.racket, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, restitution=0.95 )#物体の性質
      p.changeDynamics(self.racket, 0, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, restitution=0.95 )
      p.changeDynamics(self.racket, 1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, restitution=0.95 )
      p.changeDynamics(self.racket, 2, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, restitution=0.95 )
      p.changeDynamics(self.ball, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, restitution=0.95 )
      p.resetJointState(self.racket, 1, self.x_threshold)#壁の位置
      p.resetJointState(self.racket, 2, -self.x_threshold)
      p.setJointMotorControl2(self.racket, 0, p.VELOCITY_CONTROL, force=0)#ラケットの動作

      p.setGravity(0, 0, -self.gravity)
      p.setTimeStep(0.02)
      p.setRealTimeSimulation(0)
    
    p = self._p
    p.resetBasePositionAndOrientation(self.ball, self.ballInitPosition, p.getQuaternionFromEuler([0,0,0]))
    p.resetBaseVelocity(self.ball, self.ballInitVelocity)
    p.resetJointState(self.racket, 0, 0, 0)
    self.state = [0, 0, self.ballInitPosition[0], self.ballInitPosition[2], self.ballInitVelocity[0]]

    return np.array(self.state)
#表示
  def render(self, mode='human', close=False):
    if mode == "human":
      self._renders = True
    if mode != "rgb_array":
      return np.array([])
    base_pos=[0,0,0]
    self._render_height = 200
    self._render_width = 320
    self._cam_dist = 2
    self._cam_pitch = 0.3
    self._cam_yaw = 0
    if (self._physics_client_id>=0):
      view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
      proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
             aspect=float(self._render_width) /
             self._render_height,
             nearVal=0.1,
             farVal=100.0)
      (_, _, px, _, _) = self._p.getCameraImage(
          width=self._render_width,
          height=self._render_height,
          renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
          viewMatrix=view_matrix,
          projectionMatrix=proj_matrix)
    else:
      px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _configure(self, display=None):
    self.display = display

  def configure(self, args):
    pass
    
  def close(self):
    if self._physics_client_id >= 0:
      self._p.disconnect()
    self._physics_client_id = -1

