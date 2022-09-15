'''
石取りゲーム：人間と対戦用プログラム（DNN，DQNを利用）
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
    self.turn = BLACK
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

def main():
#環境の設定
  env_py = Board()
  env = tf_py_environment.TFPyEnvironment(env_py)
  ### ここからゲームスタート ###
  print('=== 石取りゲーム ===')
  you = input('先攻（1） or 後攻（2）を選択：')
  you = int(you)
  assert(you == BLACK or you == WHITE)

  if you == BLACK:
      adversary = WHITE
      adversary_policy_dir = f'policy_white'
  else:
      adversary = BLACK
      adversary_policy_dir = f'policy_black'
      
  policy = tf.compat.v2.saved_model.load(adversary_policy_dir) 

  print(f'ゲームスタート！')
  env_py.reset()
  while not env.game_end:
    print(f'残り{SIZE-env_py._bn}本です．')
    if env_py.turn == adversary:
      current_time_step = env_py.current_time_step()
      policy_step = policy.action( current_time_step )
      env_py.step(policy_step.action)
      print(f'{int(policy_step.action.numpy())+1}本取りました．')
    else:
      pos = input('何本取りますか？ ("1- 3")：')
      env_py.step(int(pos)-1)
      print(f'{int(pos)-1}本取りました．')
    env.change_turn()

  if env_py.turn == adversary:
    print("あなたの負け")
  else:
    print("あなたの勝ち")

if __name__ == '__main__':
  main()