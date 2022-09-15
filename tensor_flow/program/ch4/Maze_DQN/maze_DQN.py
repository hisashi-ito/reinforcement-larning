'''
迷路問題のDQNプログラム
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
#シミュレータクラスの設定
class EnvironmentSimulator(py_environment.PyEnvironment):
  def __init__(self):
    super(EnvironmentSimulator, self).__init__()
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(7, 7, 1), dtype=np.float32, minimum=0, maximum=3
    )
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=3
    )
    a = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
    self._maze = a[:,:, np.newaxis]
    self._reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#初期化
  def _reset(self):
#    print('rest')
    self._state = [1, 1]
    return ts.restart(np.array(self._maze, dtype=np.float32))
#行動による状態変化
  def _step(self, action):
    reward = 0
    import copy
    self._state_old = copy.copy(self._state)
    if action == 0:#上
      self._state[0] =  self._state[0] - 1
    elif action == 1:#右
      self._state[1] =  self._state[1] + 1
    elif action == 2:#下
      self._state[0] =  self._state[0] + 1
    else:#左
      self._state[1] =  self._state[1] - 1
    b = self._maze[self._state[0], self._state[1]]
    if b == 0:
      reward = -1
    elif b == 1:
      reward = 0
    elif b == 2:
      reward = 100
    _maze_state = self._maze.copy()
    _maze_state[self._state[0], self._state[1]] = 3
    if reward == 0:
      return ts.transition(np.array(_maze_state, dtype=np.float32), reward=reward, discount=1)
    else:
      self._state = [1, 1]
      return ts.termination(np.array(_maze_state, dtype=np.float32), reward=reward)
#ネットワーククラスの設定
class MyQNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=2, name='QNetwork'):
    super(MyQNetwork,self).__init__(
      input_tensor_spec=observation_spec, 
      state_spec=(), 
      name=name
    )
    n_action = action_spec.maximum - action_spec.minimum + 1
    self.model = keras.Sequential(
      [
        keras.layers.Conv2D(16, 3, 1, activation='relu', padding='same'),  #畳み込み
        keras.layers.Conv2D(64, 3, 1, activation='relu', padding='same'),  #畳み込み
        keras.layers.Flatten(),  #平坦化
        keras.layers.Dense(n_action),  #全結合層
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
  primary_network = MyQNetwork( env.observation_spec(), env.action_spec())
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
    max_length=10**6
  )
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    sample_batch_size=32,
    num_steps=n_step_update+1
  ).prefetch(tf.data.experimental.AUTOTUNE)
  iterator = iter(dataset)
#事前データの設定
  env.reset()
  driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env, 
    policy, 
    observers=[replay_buffer.add_batch], 
    num_episodes = 10,
  )
  driver.run(maximum_iterations=100)
  
  num_episodes = 1000
  epsilon = np.linspace(start=0.2, stop=0.0, num=num_episodes+1)#ε-greedy法用
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定
  
  for episode in range(num_episodes):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
    policy._epsilon = epsilon[episode]#エピソードに合わせたランダム行動の確率
    time_step = env.reset()#環境の初期化
  
    for t in range(100):
      policy_step = policy.action(time_step)#状態から行動の決定
      next_time_step = env.step(policy_step.action)#行動による状態の遷移

      traj =  trajectory.from_transition(time_step, policy_step, next_time_step)#データの生成
      replay_buffer.add_batch(traj)#データの保存

      experience, _ = next(iterator)#学習用データの呼び出し   
      loss_info = agent.train(experience=experience)#学習

      R = next_time_step.reward.numpy().astype('int').tolist()[0]
      episode_average_loss.append(loss_info.loss.numpy())
      episode_rewards += R
      
      if next_time_step.is_last()[0]:
        break

      time_step = next_time_step#次の状態を今の状態に設定

    print(f'Episode:{episode:4.0f}, Step:{t:3.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')


  tf_policy_saver.save(export_dir='policy')

#移動できているかのチェック
  state = [1,1]
  maze = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
  time_step = env.reset()
  for t in range(100):  #試行数分繰り返す
    maze[state[0], state[1]]=3
    policy_step = policy.action(time_step)
    time_step = env.step(policy_step.action)
    action = policy_step.action.numpy().tolist()[0]
    print(t, action, state)
    if action == 0:#上
      state[0] = state[0] - 1
    elif action == 1:#右
      state[1] = state[1] + 1
    elif action == 2:#下
      state[0] = state[0] + 1
    else:#左
      state[1] = state[1] - 1
    if maze[state[0], state[1]]==2:
      break
    if time_step.is_last()[0]:
      break
  print(maze)

if __name__ == '__main__':
  main()