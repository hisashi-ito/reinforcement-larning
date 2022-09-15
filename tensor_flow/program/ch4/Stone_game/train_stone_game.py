'''
石取りゲーム：エージェント学習プログラム（DNN，DQNを利用）
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
from tf_agents.trajectories import trajectory, policy_step as ps
from tf_agents.specs import array_spec
from tf_agents.utils import common, nest_utils

import numpy as np
import random
import copy

SIZE = 9   # 石の数
BLACK = 1  # 黒の名前
WHITE = 2  # 白の名前
REWARD_WIN = 1 # 勝ったときの報酬
REWARD_LOSE = -1 # 負けたときの報酬
#シミュレータークラス
class Board(py_environment.PyEnvironment):
  def __init__(self):
    super(Board, self).__init__()  
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(SIZE,),  dtype=np.int32, minimum=0, maximum=1
    )
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=2
    )
    self._reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#初期化
  def _reset(self):
    self._board = np.zeros((SIZE), dtype=np.int32)
    self._bn = 0
    self.winner = None 
    self.turn = random.choice([WHITE,BLACK])
    self.game_end = False # ゲーム終了チェックフラグ
    time_step = ts.restart(self._board.copy())
    return nest_utils.batch_nested_array(time_step)
#行動による状態変化
  def _step(self, pos):
    pos = nest_utils.unbatch_nested_array(pos)
    self._bn = self._bn + pos + 1
    if self._bn >= SIZE:
      self.game_end = True
      self.winner = WHITE if self.turn == BLACK else BLACK
      self._bn = SIZE
      self._board[0:self._bn] = 1
      time_step = ts.termination(self._board.copy(), reward=0)
    else:
      self._board[0:self._bn] = 1
      time_step = ts.transition(self._board.copy(), reward=0, discount=1)
    return nest_utils.batch_nested_array(time_step)
#手番の交代
  def change_turn(self):
    self.turn = WHITE if self.turn == BLACK else BLACK
  @property
  def batched(self):
    return True
  @property
  def batch_size(self):
    return 1
#必勝法通りかチェックするためのメソッド
  def check(self, pos):
    self._board = np.zeros((SIZE), dtype=np.int32)
    self._board[0:pos]=1
    time_step = ts.restart(self._board)
    return nest_utils.batch_nested_tensors(time_step)
#ネットワーククラスの設定
class MyQNetwork(network.Network):
  def __init__(self, observation_spec, action_spec, n_hidden_channels=256, name='QNetwork'):
    super(MyQNetwork,self).__init__(
      input_tensor_spec=observation_spec, 
      state_spec=(), 
      name=name
    )
    n_action = action_spec.maximum - action_spec.minimum + 1
    self.model = keras.Sequential(
      [
        keras.layers.Dense(n_hidden_channels, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(n_hidden_channels, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(n_action, kernel_initializer='he_normal'),
      ]
    )
  def call(self, observation, step_type=None, network_state=(), training=True):
    actions = self.model(observation, training=training)
    return actions, network_state

def main():
#環境の設定
  env_py = Board()
  env = tf_py_environment.TFPyEnvironment(env_py)
#黒と白の2つを宣言するために先に宣言
  primary_network = {}
  agent = {}
  replay_buffer = {}
  iterator = {}
  policy = {}
  tf_policy_saver = {}

  n_step_update = 1
  for role in [BLACK, WHITE]:#黒と白のそれぞれの設定
#ネットワークの設定
    primary_network[role] = MyQNetwork(env.observation_spec(), env.action_spec())
#エージェントの設定
    agent[role] = dqn_agent.DqnAgent(
      env.time_step_spec(),
      env.action_spec(),
      q_network = primary_network[role],
      optimizer = keras.optimizers.Adam(learning_rate=1e-3,epsilon=1e-7),
      n_step_update = n_step_update,
      target_update_period=100,
      gamma=0.99,
      train_step_counter = tf.Variable(0)
    )
    agent[role].initialize()
    agent[role].train = common.function(agent[role].train)
#行動の設定
    policy[role] = agent[role].collect_policy
#データの保存の設定
    replay_buffer[role] = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent[role].collect_data_spec,
      batch_size=env.batch_size,
      max_length=10**5
    )
    dataset = replay_buffer[role].as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        sample_batch_size=128,
        num_steps=n_step_update+1
      ).prefetch(tf.data.experimental.AUTOTUNE)
    iterator[role] = iter(dataset)
#ポリシーの保存設定
    tf_policy_saver[role] = policy_saver.PolicySaver(agent[role].policy)

  num_episodes = 2000
  epsilon = np.concatenate([np.linspace(start=1.0, stop=0.0, num=1600), np.zeros((400,)),],0)

  action_step_counter = 0
  replay_start_size = 100

  winner_counter = {BLACK:0, WHITE:0}#黒と白の勝った回数
  episode_average_loss = {BLACK:[], WHITE:[]}#黒と白の平均loss
  for episode in range(1, num_episodes + 1):
    policy[WHITE]._epsilon = epsilon[episode-1]#ε-greedy法用
    policy[BLACK]._epsilon = epsilon[episode-1]
    env.reset()

    rewards = {BLACK:0, WHITE:0}# 報酬リセット
    previous_time_step = {BLACK:None, WHITE:None}
    previous_policy_step = {BLACK:None, WHITE:None}

    while not env.game_end: # ゲームが終わるまで繰り返す
      current_time_step = copy.deepcopy(env.current_time_step())
      if previous_time_step[env.turn] is None:#1手目は学習データを作らない
        pass
      else:
        traj = trajectory.from_transition( previous_time_step[env.turn], previous_policy_step[env.turn], current_time_step )#データの生成
        replay_buffer[env.turn].add_batch( traj )#データの保存
        if action_step_counter >= 2*replay_start_size:#事前データ作成用
          experience, _ = next(iterator[env.turn])
          loss_info = agent[env.turn].train(experience=experience)#学習 
          episode_average_loss[env.turn].append(loss_info.loss.numpy())
        else:
          action_step_counter += 1

      policy_step = policy[env.turn].action(current_time_step)#状態から行動の決定
      _ = env.step(policy_step.action)#行動による状態の遷移

      previous_time_step[env.turn] = current_time_step#1つ前の状態の保存
      previous_policy_step[env.turn] = policy_step#1つ前の行動の保存

      if env.game_end:#ゲーム終了時の処理
        if env.winner == BLACK:#黒が勝った場合
          rewards[BLACK] = REWARD_WIN# 黒の勝ち報酬
          rewards[WHITE] = REWARD_LOSE# 白の負け報酬
          winner_counter[BLACK] += 1
        else:#白が勝った場合
          rewards[WHITE] = REWARD_WIN
          rewards[BLACK] = REWARD_LOSE
          winner_counter[WHITE] += 1        
        #エピソードを終了して学習
        final_time_step = env.current_time_step()#最後の状態の呼び出し
        for role in [WHITE, BLACK]:
          final_time_step = final_time_step._replace(step_type = tf.constant([2], dtype=tf.int32), reward = tf.constant([rewards[role]], dtype=tf.float32),)#最後の状態の報酬の変更
          traj = trajectory.from_transition( previous_time_step[role], previous_policy_step[role], final_time_step )#データの生成
          replay_buffer[role].add_batch( traj )#事前データ作成用
          if action_step_counter >= 2*replay_start_size:
            experience, _ = next(iterator[role])
            loss_info = agent[role].train(experience=experience)
            episode_average_loss[role].append(loss_info.loss.numpy())
      else:
        env.change_turn()

    # 学習の進捗表示 (100エピソードごと)
    if episode % 100 == 0:
      print(f'==== Episode {episode}: black win {winner_counter[BLACK]}, white win {winner_counter[WHITE]} ====')
      if len(episode_average_loss[BLACK]) == 0:
        episode_average_loss[BLACK].append(0)
      print(f'<BLACK> AL: {np.mean(episode_average_loss[BLACK]):.4f}, PE:{policy[BLACK]._epsilon:.6f}')
      if len(episode_average_loss[WHITE]) == 0:
        episode_average_loss[WHITE].append(0)
      print(f'<WHITE> AL:{np.mean(episode_average_loss[WHITE]):.4f}, PE:{policy[WHITE]._epsilon:.6f}')
      # カウンタ変数の初期化      
      winner_counter = {BLACK:0, WHITE:0}
      episode_average_loss = {BLACK:[], WHITE:[]}

  tf_policy_saver[WHITE].save(f'policy_white')
  tf_policy_saver[BLACK].save(f'policy_black')

#必勝法チェック
  for role in [WHITE, BLACK]:
    print(role)
    for i in range(0,SIZE):
      current_time_step = env.check(i)
      policy_step = agent[role].collect_policy.action(current_time_step)
      print('残り本数',SIZE-i,'取る数',policy_step.action.numpy().tolist()[0]+1,'必勝法',(SIZE-i-1)%4,'なんでもよい' if (SIZE-i-1)%4 == 0 else ('正解' if (SIZE-i-1)%4 == policy_step.action.numpy().tolist()[0]+1 else '不正解'))

if __name__ == '__main__':
  main()
