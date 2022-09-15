from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import math
import numpy as np
import time
import subprocess
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

class BulldozerBulletEnv(py_environment.PyEnvironment):
  def __init__(self, display=False):
    super(BulldozerBulletEnv, self).__init__()

    self.gravity = 9.8
    self.ballInitPosition = (0, 0.4, 0.1)
    self.boxInitPosition = (0, 1.5, 0.2)
    self.boxInitOrientation = p.getQuaternionFromEuler([0, 0, 0])
    self.forceMag = 5.0
    self.distanceThreshold = 0.1
    self.boundary = 2
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(5,),
      dtype=np.float32,
      minimum=[-self.boundary, -self.boundary, -self.boundary, -self.boundary,np.finfo(np.float32).min],
      maximum=[self.boundary, self.boundary, self.boundary, self.boundary,np.finfo(np.float32).max],
    )
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=3
    )
    self._physics_client_id = -1
    self.__lastDistance = 0
    self._reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#初期化
  def _reset(self):
    if self._physics_client_id < 0:
      self._physics_client_id = p.connect(p.GUI)
      p.resetSimulation()

      self.plane = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
      self.ball = p.loadURDF("ball.urdf", basePosition=self.ballInitPosition)  
      self.cartpole = p.loadURDF("bulldozer.urdf", basePosition=self.boxInitPosition)
      p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0)
      p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0)
      p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0)
      p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL , force=0)
      p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL , force=0)

      p.setGravity(0, 0, -self.gravity)
      p.setTimeStep(0.05)
      p.setRealTimeSimulation(0)
    
    p.resetBasePositionAndOrientation(self.ball, self.ballInitPosition, p.getQuaternionFromEuler([0,0,0]))
    p.resetBaseVelocity(self.ball, (0,0,0))

    p.resetBasePositionAndOrientation(self.cartpole, self.boxInitPosition, self.boxInitOrientation)
    p.resetBaseVelocity(self.cartpole, (0,0,0))

    self._episode_end = False

    self._state = [ self.ballInitPosition[0], self.ballInitPosition[1], self.boxInitPosition[0], self.boxInitPosition[1], self.boxInitOrientation[2] ]
    time_step = ts.restart(np.array(self._state, dtype=np.float32))

    return time_step
#行動による状態変化
  def _step(self, action):
    if self._episode_end is True:
      return self.reset()

    if action == 0:
      force = (self.forceMag, self.forceMag)
    elif action == 1:
      force = (self.forceMag, -self.forceMag)
    elif action == 2:
      force = (-self.forceMag, self.forceMag)
    elif action == 3:
      force = (-self.forceMag, -self.forceMag)
    else:
      raise Exception(f"<action> should be 0,1,2,3 but got: {action}.")
    p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force[0])
    p.setJointMotorControl2(self.cartpole, 1, p.TORQUE_CONTROL, force=force[1])
    p.stepSimulation()

    ball_pos, ball_ori = p.getBasePositionAndOrientation(self.ball)
    ball_x, ball_y, ball_z = ball_pos
    box_pos, box_ori = p.getBasePositionAndOrientation(self.cartpole)
    box_x, box_y, box_z = box_pos
    box_ori_eular = p.getEulerFromQuaternion(box_ori)

    self._state = [ ball_x, ball_y, box_x, box_y, box_ori_eular[2] ]

    if   box_x > self.boundary or box_x < -self.boundary  \
      or box_y > self.boundary or box_y < -self.boundary  \
      or ball_x > self.boundary or ball_x < -self.boundary  \
      or ball_y > self.boundary or ball_y < -self.boundary:
      self._episode_end = True
      time_step = ts.termination(np.array(self._state, dtype=np.float32), reward=-1)
    elif ball_x * ball_x + ball_y * ball_y < self.distanceThreshold * self.distanceThreshold:
      self._episode_end = True
      time_step = ts.termination(np.array(self._state, dtype=np.float32), reward=1) 
    else:
      time_step = ts.transition(np.array(self._state, dtype=np.float32), reward=0, discount=1)

    return time_step
    
  def close(self):
    if self._physics_client_id >= 0:
      p.disconnect()
    self._physics_client_id = -1
