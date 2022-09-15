'''
リバーシプログラム：人間と対戦用プログラム（CNN，DQNを利用）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import numpy as np
import random

import tensorflow as tf

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils

import re

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

SIZE = 4 # 盤面のサイズ SIZE*SIZE
NONE = 0 # 盤面のある座標にある石：なし
BLACK = 1# 盤面のある座標にある石：黒
WHITE = 2# 盤面のある座標にある石：白
STONE = {NONE:" ", BLACK:"●", WHITE:"○"}# 石の表示用
ROWLABEL = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8} #ボードの横軸ラベル
N2L = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] # 横軸ラベルの逆引き用
REWARD_WIN = 1 # 勝ったときの報酬
REWARD_LOSE = -1 # 負けたときの報酬
# 2次元のボード上での隣接8方向の定義（左から，上，右上，右，右下，下，左下，左，左上）
DIR = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1, -1), (0,-1), (-1,-1))
#シミュレータークラス
class Board(py_environment.PyEnvironment):  
  def __init__(self):
    super(Board, self).__init__()  
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(SIZE,SIZE,1), dtype=np.float32, minimum=0, maximum=2
    )
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=SIZE*SIZE-1
    )
    self.reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#ボードの初期化
  def _reset(self):
    self.board = np.zeros((SIZE, SIZE, 1), dtype=np.float32) # 全ての石をクリア．ボードは2次元配列（i, j）で定義する．
    mid = SIZE // 2 # 真ん中の基準ポジション
    # 初期4つの石を配置
    self.board[mid, mid] = WHITE
    self.board[mid-1, mid-1] = WHITE
    self.board[mid-1, mid] = BLACK
    self.board[mid, mid-1] = BLACK
    self.winner = NONE # 勝者
    self.turn = random.choice([BLACK,WHITE])
    self.game_end = False # ゲーム終了チェックフラグ
    self.pss = 0 # パスチェック用フラグ．双方がパスをするとゲーム終了
    self.nofb = 0 # ボード上の黒石の数
    self.nofw = 0 # ボード上の白石の数
    self.available_pos = self.search_positions() # self.turnの石が置ける場所のリスト

#    time_step = ts.restart(self.board.flatten())
    time_step = ts.restart(self.board)
    return nest_utils.batch_nested_array(time_step)
#行動による状態変化（石を置く&リバース処理）
  def _step(self, pos):
    pos = nest_utils.unbatch_nested_array(pos)
    pos = divmod(pos, SIZE)     
    if self.is_available(pos):
      self.board[pos[0], pos[1]] = self.turn
      self.do_reverse(pos) # リバース
    self.end_check()#終了したかチェック
    time_step = ts.transition(self.board, reward=0, discount=1)
#    time_step = ts.transition(self.board.flatten(), reward=0, discount=1)
    return nest_utils.batch_nested_array(time_step)
#ターンチェンジ
  def change_turn(self, role=None):
    if role is NONE:
      role = random.choice([WHITE,BLACK])
    if role is None or role != self.turn:
      self.turn = WHITE if self.turn == BLACK else BLACK
      self.available_pos = self.search_positions() # 石が置ける場所を探索しておく
#ランダムに石を置く場所を決める（ε-greedy用）
  def random_action(self):
    if len(self.available_pos) > 0:
      pos = random.choice(self.available_pos) # 置く場所をランダムに決める
      pos = pos[0] * SIZE + pos[1] # 1次元座標に変換（NNの教師データは1次元でないといけない）
      return pos
    return False # 置く場所なし
#リバース処理
  def do_reverse(self, pos):
    for di, dj in DIR:
      opp = BLACK if self.turn == WHITE else WHITE # 対戦相手の石
      boardcopy = self.board.copy() # 一旦ボードをコピーする（copyを使わないと参照渡しになるので注意）
      i = pos[0]
      j = pos[1]
      flag = False # 挟み判定用フラグ
      while 0 <= i < SIZE and 0 <= j < SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
        i += di # i座標（縦）をずらす
        j += dj # j座標（横）をずらす
        if 0 <= i < SIZE and 0 <= j < SIZE and boardcopy[i,j] == opp:  # 盤面に収まっており，かつ相手の石だったら
          flag = True
          boardcopy[i,j] = self.turn # 自分の石にひっくり返す
        elif not(0 <= i < SIZE and 0 <= j < SIZE) or (flag == False and boardcopy[i,j] != opp):
          break
        elif boardcopy[i,j] == self.turn and flag == True: # 自分と同じ色の石がくれば挟んでいるのでリバース処理を確定
          self.board = boardcopy.copy() # ボードを更新
          break

#石が置ける場所をリストアップする．石が置ける場所がなければ「パス」となる
  def search_positions(self):
    pos = []
    emp = np.where(self.board == 0) # 石が置かれていない場所を取得
    for i in range(emp[0].size): # 石が置かれていない全ての座標に対して
      p = (emp[0][i], emp[1][i]) # (i,j)座標に変換
      if self.is_available(p):
        pos.append(p) # 石が置ける場所の座標リストの生成
    return pos
#石が置けるかをチェックする
  def is_available(self, pos):
    if self.board[pos[0], pos[1]] != NONE: # 既に石が置いてあれば，置けない
      return False
    opp = BLACK if self.turn == WHITE else WHITE
    for di, dj in DIR: # 8方向の挟み（リバースできるか）チェック
      i = pos[0]
      j = pos[1]
      flag = False # 挟み判定用フラグ
      while 0 <= i < SIZE and 0 <= j < SIZE: # (i,j)座標が盤面内に収まっている間繰り返す
        i += di # i座標（縦）をずらす
        j += dj # j座標（横）をずらす
        if 0 <= i < SIZE and 0 <= j < SIZE and self.board[i,j] == opp: #盤面に収まっており，かつ相手の石だったら
          flag = True
        elif not(0 <= i < SIZE and 0 <= j < SIZE) or (flag == False and self.board[i,j] != opp) or self.board[i,j] == NONE:        
          break
        elif self.board[i,j] == self.turn and flag == True: # 自分と同じ色の石          
          return True
    return False
    
#ゲーム終了チェック
  def end_check(self):
    if np.count_nonzero(self.board) == SIZE * SIZE or self.pss == 2: # ボードに全て石が埋まるか，双方がパスがしたら
      self.game_end = True
      self.nofb = len(np.where(self.board==BLACK)[0])
      self.nofw = len(np.where(self.board==WHITE)[0])
      if self.nofb > self.nofw:
        self.winner = BLACK
      elif self.nofb < self.nofw:
        self.winner = WHITE
      else:
        self.winner = NONE
#ボードの表示（人間との対戦用）
  def show_board(self):
    print('  ', end='')      
    for i in range(1, SIZE + 1):
      print(f' {N2L[i]}', end='') # 横軸ラベル表示
    print('')
    for i in range(0, SIZE):
      print(f'{i+1:2d} ', end='')
      for j in range(0, SIZE):
        print(f'{STONE[int(self.board[i][j])]} ', end='') 
      print('')
#パスしたときの処理  
  def add_pass(self):
    self.pss += 1
#パスした後の処理  
  def clear_pass(self):
    self.pss = 0
  
  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return 1

def convert_coordinate(pos):
  pos = pos.split(' ')
  i = int(pos[0]) - 1
  j = int(ROWLABEL[pos[1]]) - 1
  return i*SIZE + j

def judge(board, a, you):
  if board.winner == a:
    print('Game over. You lose!')
  elif board.winner == you:
    print('Game over. You win！')
  else:
    print('Game over. Draw.')

def main():
#環境の設定
  board = Board()
  ### ここからゲームスタート ###
  print('=== リバーシ ===')
  you = input('先攻（黒石, 1） or 後攻（白石, 2）を選択：')
  you = int(you)
  assert(you == BLACK or you == WHITE)

  level = input('難易度（弱 1〜10 強）：')
  level = int(level) * 20

  if you == BLACK:#ポリシーの読み込み
    adversary = WHITE
    adversary_policy_dir = f'policy_white_{level}'
    stone = '「●」（先攻）'    
  else:
    adversary = BLACK
    adversary_policy_dir = f'policy_black_{level}'
    stone = '「○」（後攻）'
    
  policy = tf.compat.v2.saved_model.load(adversary_policy_dir) 

  print(f'あなたは{stone}です。ゲームスタート！')
  board.reset()
  board.change_turn(BLACK)
  board.show_board()
  # ゲーム開始
  while not board.game_end:
#エージェントの手番
    if board.turn == adversary:
      current_time_step = board.current_time_step()
      action_step = policy.action( current_time_step )
      pos = int(action_step.action.numpy())
      if not board.is_available(divmod(pos,SIZE)): # NNで置く場所が置けない場所であれば置ける場所からランダムに選択する．
        pos = board.random_action()
        if pos is False: # 置く場所がなければパス         
          board.add_pass()

      print('エージェントのターン --> ', end='')
      if board.pss > 0 and pos is False:
        print(f'パスします。{board.pss}')
      else:
        board.step(pos) # posに石を置く
        board.clear_pass()
        pos = divmod(pos, SIZE)
        print(f'({pos[0]+1},{N2L[pos[1]+1]})')
      board.show_board()
      board.end_check() # ゲーム終了チェック
      if board.game_end:
        judge(board, adversary, you)
        continue
      board.change_turn() #　エージェント --> You
#プレーヤーの手番
    while True:
      print('あなたのターン。')
      if not board.search_positions():
        print('パスします。')
        board.add_pass()
      else:
        pos = input('どこに石を置きますか？ (行列で指定。例 "4 d")：')
        if not re.match(r'[0-9] [a-z]', pos):
          print('正しく座標を入力してください。')
          continue
        else:
          pos = convert_coordinate(pos)
          if not board.is_available(divmod(pos,SIZE)): # 置けない場所に置いた場合
            print('ここには石を置けません。')
            continue
          board.step(pos)
          board.show_board()
          board.clear_pass()
      break
#手番の変更
    board.end_check()
#終了判定
    if board.game_end:
      judge(board, adversary, you)
      continue

    board.change_turn() 

if __name__ == '__main__':
  main()
