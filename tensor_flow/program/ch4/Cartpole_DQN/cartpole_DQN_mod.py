#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.agents.dqn import dqn_agent # DQN のAgentがサポートされている
from tf_agents.networks import network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver

import gym
import numpy as np
import time

class QNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, n_hidden_channels=50, name='QNetwork'):
        super(QNetwork, self).__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            name=name
        )
        
        # 倒立振子のactionは[0,1] の2個 
        n_action = action_spec.maximum - action_spec.minimum + 1
        
        # モデルの設定
        # keras はinput size 指定なしでO.K.
        # 活性化関数は深層強化学習では一般的に`tanh`を指定することが多い
        # n_action はアクションのアクション数
        self.model = keras.Sequential(
            [
                keras.layers.Dense(n_hidden_channels, activation='tanh'),
                keras.layers.Dense(n_hidden_channels, activation='tanh'),
                keras.layers.Dense(n_action)
            ]
        )

    def call(self, observation, step_type=None, network_state=(), training=True):
        # keras は __call__ じゃなくてcallでfoward を計算
        actions = self.model(observation, training=training)
        return actions, network_state

def main():
    # Gym の環境をGymWrapper経由でTFPyEnvironmentへ渡す
    env_py = gym.make('CartPole-v0')
    env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env_py))

    # policy model の入出力sizeを設定してモデルを構築
    policy_network = QNetwork(
        env.observation_spec(),
        env.action_spec()
    )    

    n_step_update = 1

    agent = dqn_agent.DqnAgent(
        env.time_step_spec()
        env.action_spec(),
        q_network=policy_network,
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3, epsilon=1e-2
        ),
        n_step_update=n_step_update, # Q ネットワークを更新する頻度
        epsilon_greedy=1.0,
        target_update_tau=1.0,       # Q ネットワークを更新する頻度を設定する係数
        target_update_period=100,    # Q ネットワークを更新する場合に、設定したステップ数前のネットワークを使って更新 
        gamma=0.99,                  # 割引率、学習率はない
        td_errors_loss_fn = common.element_wise_squared_loss, # \big( r_{tL1} + \gamma \rm{max}_{p} q(s_{t+1}, p) - Q (s_{t}, p)  \big)^2
        train_step_counter = tf.Variable(0)
    )

    agent.initialize()
    agent.train = common.function(agent.train)

    # 行動の設定
    policy = agent.collect_policy
    
    # データの保存設定
    # 保存しているすべてのデータが等しい確率で取り出さえるように設定する
    # `TFUniformReplayBuffer`
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
    
    # 事前データの設定
    env.reset()

    # ドライバの設定
    # {行動前の状態, 行動, 行動後の状態} をnum_episodesで
    # 設定した回数の行動を繰り返し行って集めていく
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        policy,
        observers=[replay_buffer.add_batch],
        num_episodes = 100
    )
    driver.run(maximum_iterations=10000)

    num_episodes = 200

    # ポリシーの保存設定
    tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)
    
    for episode in range(num_episodes+1):
        episode_rewards = 0
        episode_average_loss = []

        # epsilon の設定
        # 最初は大きく徐々に0.5に近くなる
        policy._epsilon = 0.5 * (1 / (episode + 1))

        # 環境の初期値を設定
        time_step = env.reset()
    
        while True:
            if episode % 10 == 0:
                env_py.render('human')

            policy_step = policy.action(time_step)
            next_time_step = env.step(policy_step.action)

            # {行動前の状態, 行動, 行動後の状態}をreplay_bufferへ保存
            traj =  trajectory.from_transition(time_step, policy_step, next_time_step)
            replay_buffer.add_batch(traj)

            # 学習用データの呼び出し
            experience, _ = next(iterator)
            
            # 学習
            loss_info = agent.train(experience=experience)
            
            # 報酬値
            R = next_time_step.reward.numpy().astype('int').tolist()[0]
            
            # 出力用のlossを保存
            episode_average_loss.append(loss_info.loss.numpy())
            episode_rewards += R
            
            # 状態の遷移
            time_step = next_time_step
            
            if next_time_step.is_last():
                # 棒が倒れてしまったら終了
                break
            
            if episode_rewards >= 200:
                # タスクが完了(トータルの報酬が200以上)
                break

    # モデルを保存
    tf_policy_saver.save(export_dir='policy')
    env.close()


if __name__ == '__main__':
  main()
