'''
RaspberryPi用
ネズミ学習問題のDQNプログラム（全部）
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
#ネットワークの設定
class MyQNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=2, name='QNetwork'):
    super(MyQNetwork, self).__init__(
      input_tensor_spec=observation_spec, 
      state_spec=(), 
      name=name
    )
    n_action = action_spec.maximum - action_spec.minimum + 1
    #ネットワークの設定
    self.model = keras.Sequential(
      [
        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),#畳み込み
        keras.layers.MaxPool2D(pool_size=(2, 2)),#プーリング
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),#畳み込み
        keras.layers.MaxPool2D(pool_size=(2, 2)),#プーリング
        keras.layers.Flatten(),#平坦化
        keras.layers.Dense(2, activation='softmax'),#全結合層
      ]
    )
  def call(self, observation, step_type=None, network_state=(), training=True):
    observation = tf.cast(observation, tf.float64)
    actions = self.model(observation, training=training)
    return actions, network_state

def main():
#環境の設定
  env_py = EnvironmentSimulator()
  env = tf_py_environment.TFPyEnvironment(env_py)
#ネットワークの設定
  primary_network = MyQNetwork(env.observation_spec(), env.action_spec())
#エージェントの設定
  n_step_update = 1
  agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=primary_network,
    optimizer=keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-2),
    n_step_update=n_step_update,
    epsilon_greedy=1.0,
    target_update_tau=1.0,
    target_update_period=10,
    gamma=0.9,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter = tf.Variable(0)
  )
  agent.initialize()
  agent.train = common.function(agent.train)
#行動の設定
  policy = agent.collect_policy
#データの保存の設定
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=10**3
  )
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    sample_batch_size=32,
    num_steps=n_step_update+1
  ).prefetch(tf.data.experimental.AUTOTUNE)
  iterator = iter(dataset)
#事前データの設定
  env.reset()
  driver = dynamic_step_driver.DynamicStepDriver(
    env, 
    policy, 
    observers=[replay_buffer.add_batch], 
    num_steps = 10,
  )
  driver.run(maximum_iterations=100)
  
  num_episodes = 100
  epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes+1)#ε-greedy法用
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定

  for episode in range(num_episodes+1):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
    policy._epsilon = epsilon[episode]#エピソードに合わせたランダム行動の確率
    time_step = env.reset()#環境の初期化
  
    for t in range(5):#各エピソード5回の行動
      policy_step = policy.action(time_step)#状態から行動の決定
      next_time_step = env.step(policy_step.action)#行動による状態の遷移

      traj =  trajectory.from_transition(time_step, policy_step, next_time_step)#データの生成
      replay_buffer.add_batch(traj)#データの保存

      experience, _ = next(iterator)#学習用データの呼び出し   
      loss_info = agent.train(experience=experience)#学習

#      S = time_step.observation.numpy().tolist()[0]#状態
      A = policy_step.action.numpy().tolist()[0]#行動
      R = next_time_step.reward.numpy().astype('int').tolist()[0]#報酬
      print(A, R)
      episode_average_loss.append(loss_info.loss.numpy())#lossの計算
      episode_rewards += R#報酬の合計値の計算

      time_step = next_time_step#次の状態を今の状態に設定

    print(f'Episode:{episode:4.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')
    time.sleep(3)

  tf_policy_saver.save(export_dir='policy')#ポリシーの保存
  cap.release()
  env.close()
  
if __name__ == '__main__':
  main()
