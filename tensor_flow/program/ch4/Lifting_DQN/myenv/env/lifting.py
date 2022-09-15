"""
リフティングの動作プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class LiftingEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }
#初期設定
  def __init__(self):
    self.gravity = 9.8#重力加速度
    self.racketmass = 1.0#ラケット重さ
    self.racketwidth = 0.5#ラケットの横幅
    self.racketheight = 0.25#ラケットの高さ
    self.racketposition = 0#ラケットの位置
    self.ballPosition = 1#ボールの位置
    self.ballRadius = 0.1#ボールの半径
    self.ballVelocity = 1#ボールの横方向の速度
    self.force_mag = 10.0#台車を移動させるときの力
    self.tau = 0.02  # 時間刻み
    self.cx_threshold = 2.4#移動制限
    self.bx_threshold = 2.4
    self.by_threshold = 2.4

    self.action_space = spaces.Discrete(2)
    high = np.array([
      self.cx_threshold, np.finfo(np.float32).max, self.bx_threshold, self.by_threshold, np.finfo(np.float32).max
      ])
    self.observation_space = spaces.Box(-high, high)

    self._seed()
    self.viewer = None
    self._reset()
#乱数の設定
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
#行動の設定
  def _step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    state = self.state
    cx, cx_dot, bx, by, bx_dot = state
    force = self.force_mag if action==1 else -self.force_mag#行動によって力の方向を決める
    cx_dot = cx_dot + self.tau * force / self.racketmass#ラケットの速度の更新
    cx  = cx + self.tau * cx_dot#ラケットの位置の更新

    byacc  = -self.gravity#ボールの加速度
    self.by_dot = self.by_dot + self.tau * byacc#ボールのy方向の速度の更新
    by  = by + self.tau * self.by_dot#ボールのy方向の位置の更新

    bx  = bx + self.tau * bx_dot#ボールのx方向の位置の更新
    bx_dot = bx_dot if bx>-self.cx_threshold and bx<self.cx_threshold else -bx_dot#壁にぶつかったらx方向の速度を反転
    reward = 0.0
    if bx>cx-self.racketwidth/2 and bx<cx+self.racketwidth/2 and by<self.ballRadius and self.by_dot<0:#ラケットにぶつかったか？
      self.by_dot = -self.by_dot#ボールのy方向の速度を反転
      reward = 1.0
    self.state = (cx, cx_dot,bx,by,bx_dot)
    done =  cx < -self.cx_threshold-self.racketwidth or cx > self.cx_threshold +self.racketwidth or by < 0#ボールがラケットより下か？
    done = bool(done)

    if done:
      reward = 0.0

    return np.array(self.state), reward, done, {}
#初期化
  def _reset(self):
    self.state = np.array([0,0,0,self.ballPosition,self.ballVelocity])
    self.steps_beyond_done = None
    self.by_dot = 0
    return np.array(self.state)
#表示
  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400
    world_width = self.cx_threshold*2#壁位置が画面の枠と一致するように
    scale = screen_width/world_width#倍率を決める
    racketwidth = self.racketwidth*scale
    racketheight = self.racketheight*scale

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      l,r,t,b = -racketwidth/2, racketwidth/2, racketheight/2, -racketheight/2
      axleoffset =racketheight/4.0
      racket = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])#ラケットの描画
      self.rackettrans = rendering.Transform()
      racket.add_attr(self.rackettrans)
      self.viewer.add_geom(racket)
      
      ball = rendering.make_circle(0.1*scale)#ボールの描画
      self.balltrans = rendering.Transform()
      ball.add_attr(self.balltrans)
      self.viewer.add_geom(ball)
      
    if self.state is None: return None

    x = self.state
    rackety = self.racketposition*scale # ラケットのy座標
    racketx = x[0]*scale+screen_width/2.0 # ラケットのx座標
    ballx = x[2]*scale+screen_width/2.0 # ボールのx座標
    bally = x[3]*scale#ボールのy座標
    self.rackettrans.set_translation(racketx, rackety)
    self.balltrans.set_translation(ballx, bally)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')
