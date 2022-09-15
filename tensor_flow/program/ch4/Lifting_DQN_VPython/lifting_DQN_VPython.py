'''
リフティング問題のDQNプログラム（VPython使用）
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

import numpy as np
import random
import vpython as vs
#シミュレータクラスの設定
class EnvironmentSimulator(py_environment.PyEnvironment):
  def __init__(self):
    super(EnvironmentSimulator, self).__init__()
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

    high = np.array([ self.cx_threshold, np.finfo(np.float32).max, self.bx_threshold, self.by_threshold, np.finfo(np.float32).max])
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=1
    )
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(5,), dtype=np.float32, minimum=-high,  maximum=high
    )
    scene = vs.canvas(x=0, y=0, width=600, height=400, center=vs.vector(0,0,0), background=vs.vector(1,1,1))#環境の設定
    self.ball = vs.sphere(pos=vs.vector(0,self.ballPosition,0), radius=self.ballRadius, color=vs.vector(0.8,0.8,0.8), make_trail=True, retain=2000)#ボールの設定
    self.racket = vs.box(pos=vs.vector(self.racketposition,-self.racketheight/2,0), size=vs.vector(self.racketwidth,self.racketheight,0.2), color=vs.vector(0.5,0.5,0.5)) #ラケットの設定
    self.wallR = vs.box(pos=vs.vector(self.cx_threshold+0.2,0.5,0), size=vs.vector(0.2,1,1), color=vs.vector(0.2,0.2,0.2)) #右の壁の設定
    self.wallL = vs.box(pos=vs.vector(-self.cx_threshold-0.2,0.5,0), size=vs.vector(0.2,1,1), color=vs.vector(0.2,0.2,0.2)) #左の壁の設定

    self.viewer = None
    self._reset()
  def observation_spec(self):#追加（定型）
    return self._observation_spec
  def action_spec(self):#追加（定型）
    return self._action_spec
#行動による状態変化
  def _step(self, action):
    state = self.state
    cx, cx_dot, bx, by, bx_dot = state
    force = self.force_mag if action==1 else -self.force_mag
    cx_dot = cx_dot + self.tau * force / self.racketmass
    cx  = cx + self.tau * cx_dot

    byacc  = -self.gravity
    self.by_dot = self.by_dot + self.tau * byacc
    by  = by + self.tau * self.by_dot

    bx  = bx + self.tau * bx_dot
    bx_dot = bx_dot if bx>-self.cx_threshold and bx<self.cx_threshold else -bx_dot
    reward = 0.0
    if bx>cx-self.racketwidth/2 and bx<cx+self.racketwidth/2 and by<self.ballRadius and self.by_dot<0:
      self.by_dot = -self.by_dot
      reward = 1.0
    self.state = (cx, cx_dot,bx,by,bx_dot)
    done =  cx < -self.cx_threshold-self.racketwidth \
        or cx > self.cx_threshold +self.racketwidth\
        or by < 0
    done = bool(done)

    if done:
      reward = 0.0
      return ts.termination(np.array(self.state, dtype=np.float32), reward=reward)
    else:
      return ts.transition(np.array(self.state, dtype=np.float32), reward=reward, discount=1)
      
#初期化
  def _reset(self):
    self.state = np.array([0,0,0,self.ballPosition,self.ballVelocity])
    self.steps_beyond_done = None
    self.by_dot = 0
    self.ball.pos = vs.vector(0,self.ballPosition,0)#ボールの初期位置
    self.racket.pos = vs.vector(self.racketposition,-self.racketheight/2,0)#ラケットの初期位置
    self.ball.clear_trail()#軌跡の消去
    return ts.restart(np.array(self.state, dtype=np.float32))
#表示
  def render(self, mode='human'):
    vs.rate(20)
    cx, cx_dot,bx,by,bx_dot = self.state
    self.ball.pos = vs.vector(bx,by,0)
    self.racket.pos = vs.vector(cx,-self.racketheight/2,0)
#ネットワーククラスの設定
class MyQNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=50, name='QNetwork'):
    super(MyQNetwork, self).__init__(
      input_tensor_spec=observation_spec, 
      state_spec=(), 
      name=name
    )
    n_action = action_spec.maximum - action_spec.minimum + 1
    self.model = keras.Sequential(
      [
        keras.layers.Dense(n_hidden_channels, activation='tanh'),
        keras.layers.Dense(n_hidden_channels, activation='tanh'),
        keras.layers.Dense(n_action),
      ]
    )
  def call(self, observation, step_type=None, network_state=(), training=True):
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
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-3),
    n_step_update=n_step_update,
    epsilon_greedy=1.0,
    target_update_tau=1.0,
    target_update_period=100,
    gamma=0.99,
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
    max_length=10**6
  )
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    sample_batch_size=128,
    num_steps=n_step_update+1
  ).prefetch(tf.data.experimental.AUTOTUNE)
  iterator = iter(dataset)
#事前データの設定
  env.reset()
  driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env, 
    policy, 
    observers=[replay_buffer.add_batch], 
    num_episodes = 100,
  )
  driver.run(maximum_iterations=10000)

  num_episodes = 500
  epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes+1)
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定

  for episode in range(num_episodes+1):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
    policy._epsilon = epsilon[episode]#ランダム行動の確率
    time_step = env.reset()#環境の初期化

    while True:
      if episode%10 == 0:#10回に1回だけ描画（高速に行うため）
        env_py.render('human')

      policy_step = policy.action(time_step)#状態から行動の決定
      next_time_step = env.step(policy_step.action)#行動による状態の遷移

      traj =  trajectory.from_transition(time_step, policy_step, next_time_step)#データの生成
      replay_buffer.add_batch(traj)#データの保存

      experience, _ = next(iterator)#学習用データの呼び出し
      loss_info = agent.train(experience=experience)#学習

      R = next_time_step.reward.numpy().astype('int').tolist()[0]#報酬
      episode_average_loss.append(loss_info.loss.numpy())#lossの計算
      episode_rewards += R#報酬の合計値の計算

      time_step = next_time_step#次の状態を今の状態に設定

      if next_time_step.is_last():#終了？（ボールがラケットより下に着た場合）
        break
      if episode_rewards == 10:#報酬が10？（10回跳ね返した場合）
        break

    print(f'Episode:{episode:4.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')

  tf_policy_saver.save(export_dir='policy')#ポリシーの保存
  env.close()
  
if __name__ == '__main__':
  main()