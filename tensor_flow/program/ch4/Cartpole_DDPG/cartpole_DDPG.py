'''
倒立振子のDDPGプログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.agents.ddpg import ddpg_agent
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
from cartpole import CartPoleEnv

class ActorNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=100, name='actor'):
    super(ActorNetwork,self).__init__(
      input_tensor_spec=observation_spec, 
      state_spec=(), 
      name=name
    )
    n_action_dims = action_spec.shape[0]
    inputs = keras.Input(shape=observation_spec.shape)
    h1 = keras.layers.Dense(n_hidden_channels, activation='relu')(inputs)
    outputs = keras.layers.Dense(n_action_dims, activation="tanh")(h1)
    self.model = keras.Model(inputs=inputs, outputs=outputs)
  def call(self, observation, step_type=None, network_state=(), training=True):
    actions = self.model(observation, training=training)
    return actions, network_state

class CriticNetwork(network.Network):
  def __init__(self, input_tensor_spec, n_hidden_channels=100, name='critic'):
    super(CriticNetwork,self).__init__(
      input_tensor_spec=input_tensor_spec, 
      state_spec=(), 
      name=name
    )
    observation_spec, action_spec = input_tensor_spec
    inputs_observation = keras.Input(shape=observation_spec.shape)
    inputs_action = keras.Input(shape=action_spec.shape)
    conc = keras.layers.Concatenate(axis=1)([inputs_observation,inputs_action])
    h1 = keras.layers.Dense(n_hidden_channels, activation='relu')(conc)
    outputs = keras.layers.Dense(1)(h1)
    self.model = keras.Model(inputs=[inputs_observation,inputs_action], outputs=outputs)
  def call(self, inputs, step_type=None, network_state=(), training=True):
    q_value = self.model(inputs, training=training)
    q_value = keras.backend.flatten(q_value)
    return q_value, network_state

def main():
#環境の設定
  env_py = CartPoleEnv()
  env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env_py))
#ネットワークの設定
  my_actor_network = ActorNetwork(env.observation_spec(), env.action_spec())
  my_critic_network = CriticNetwork( (env.observation_spec(), env.action_spec()))
#エージェントの設定
  agent = ddpg_agent.DdpgAgent(
    env.time_step_spec(),
    env.action_spec(),
    actor_network = my_actor_network,
    critic_network = my_critic_network,
    actor_optimizer = keras.optimizers.Adam(0.001),
    critic_optimizer = keras.optimizers.Adam(0.001),
    target_update_tau = 0.001,
    target_update_period= 1,
    gamma=0.99,
    reward_scale_factor = 0.1,
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
    num_steps=2,
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

  num_episodes = 2000
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定

  for episode in range(num_episodes+1):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
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

      if next_time_step.is_last():#終了？（棒が倒れた場合）
        break
      if episode_rewards == 200:#報酬が200？（棒が一定時間立っていた場合）
        break

    if len(episode_average_loss) == 0:
      episode_average_loss.append(0)
    print(f'Episode:{episode:4.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}')

  tf_policy_saver.save(export_dir='policy')#ポリシーの保存
  env.close()

if __name__ == '__main__':
  main()