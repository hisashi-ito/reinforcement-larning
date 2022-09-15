'''
Open AI Gym(Atari)スペースインベーダーゲームの学習（DQN利用）
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

import pybullet as p
import numpy as np
import random
import time
from lifting_bullet import LiftingBulletEnv
#ネットワーククラスの設定
class MyQNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=50, name='QNetwork'):
    super(MyQNetwork,self).__init__(
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
  env_py = LiftingBulletEnv(renders=True)
  env = tf_py_environment.TFPyEnvironment( gym_wrapper.GymWrapper(env_py) )
#ネットワークの設定
  primary_network = MyQNetwork(env.observation_spec(),  env.action_spec())
#エージェントの設定
  n_step_update = 1
  agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=primary_network,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-2),
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

  num_episodes = 5000
  epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes+1)
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定

  for episode in range(num_episodes+1):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
    policy._epsilon = epsilon[episode]#ランダム行動の確率
    time_step = env.reset()#環境の初期化

    while True:
#      if episode%10 == 0:#10回に1回だけ描画（高速に行うため）
#        env_py.render('human')
      time.sleep(0.01)#高速に行うときには削除

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
      if episode_rewards >= 10:#報酬が10？（10回跳ね返した場合）
        break

    print(f'Episode:{episode:4.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')

  tf_policy_saver.save(export_dir='policy')#ポリシーの保存
  env.close()
  
if __name__ == '__main__':
  main()
