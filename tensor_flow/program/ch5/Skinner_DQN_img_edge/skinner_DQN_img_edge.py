'''
RaspberryPi用
ネズミ学習問題のDQNプログラム（エッジ動作）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.utils import nest_utils

import numpy as np
import random
import time
import os

#入出力の設定
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)#ピン配置の番号を使用
GPIO.setup(22, GPIO.IN)#22番ピンを入力（報酬用）
#サーボモータの設定
import Adafruit_PCA9685
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
pwm.set_pwm(0, 0, 400) # サーボモータを初期位置へ
time.sleep(1)
#カメラの設定
import cv2
cap = cv2.VideoCapture(0)
SIZE=16
#シミュレータクラスの設定
class EnvironmentSimulator(py_environment.PyEnvironment):
  def __init__(self):
    super(EnvironmentSimulator, self).__init__()
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(SIZE, SIZE, 1), dtype=np.float64, minimum=0, maximum=1
    )
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1
    )
    self._reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#初期化
  def _reset(self):
    img_state = np.zeros((SIZE, SIZE, 1), dtype=np.float64)
    return ts.restart(img_state)
#行動による状態変化
  def _step(self, action):
    reward = 0
    if action == 0:#電源ボタンを押す
      pwm.set_pwm(0, 0, 250) 
      time.sleep(1)
      if GPIO.input(22)==0:#商品があれば
        reward = 1#報酬が得られる
    else:#行動ボタンを押す
      pwm.set_pwm(0, 0, 550) 
      time.sleep(1)
      if GPIO.input(22)==0:#商品があれば
        reward = 1#報酬が得られる
    pwm.set_pwm(0, 0, 400) 
    time.sleep(1)
    ret, frame = cap.read()#画像の読み込み
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
    xp = int(frame.shape[1]/2)
    yp = int(frame.shape[0]/2)
    d = 150
    cv2.rectangle(gray, (xp-d, yp-d), (xp+d, yp+d), color=0, thickness=2)#切り抜く範囲を表示
    img = cv2.resize(gray[yp-d:yp + d, xp-d:xp + d],(SIZE, SIZE))#画像の中心を切り抜いて8×8の画像に変換
    img = img/256.0
    img_state = img.reshape(SIZE, SIZE, 1) # 3次元行列に変換（8×8×1，縦×横×チャンネル数）
    return ts.transition(img_state, reward=reward, discount=1)#TF-Agents用の戻り値の生成

def main():
#環境の設定
  env_py = EnvironmentSimulator()
  env = tf_py_environment.TFPyEnvironment(env_py)
#行動の設定
  policy = tf.compat.v2.saved_model.load(os.path.join('policy'))

  episode_rewards = 0#報酬の計算用
  policy._epsilon = 0#epsilon[episode]#エピソードに合わせたランダム行動の確率
  time_step = env.reset()#環境の初期化

  for t in range(5):#各エピソード5回の行動
    policy_step = policy.action(time_step)#状態から行動の決定
    next_time_step = env.step(policy_step.action)#行動による状態の遷移

    A = policy_step.action.numpy().tolist()[0]#行動
    R = next_time_step.reward.numpy().astype('int').tolist()[0]#報酬
    print(A, R)
    episode_rewards += R#報酬の合計値の計算

    time_step = next_time_step

  print(f'Rewards:{episode_rewards}')

if __name__ == '__main__':
  main()
