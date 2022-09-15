'''
迷路問題のQラーニングプログラム（学習済みQ値を読み込んでテスト）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import numpy as np
#シミュレータクラスの設定
class MyEnvironmentSimulator():
  def __init__(self):
    self._maze = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
    self.reset()
#初期化
  def reset(self):
    self._state = [1,1]
    return np.array(self._state)
#行動による状態変化
  def step(self, action):
    reward = 0
    if action == 0:#上
      self._state[0] = self._state[0] - 1
    elif action == 1:#右
      self._state[1] = self._state[1] + 1
    elif action == 2:#下
      self._state[0] = self._state[0] + 1
    else:#左
      self._state[1] = self._state[1] - 1
    b = self._maze[self._state[0], self._state[1]]
    if b == 0:
      reward = -1
    elif b == 1:
      reward = 0
    elif b == 2:
      reward = 1
    return np.array(self._state), reward
#Q値の設定
class MyQTable():
  def __init__(self):
    #self._Qtable = np.zeros((4, 7, 7))
    qt = np.loadtxt('Qvalue.txt')
    self._Qtable = qt.reshape(4, 7, 7)
#行動の選択
  def get_action(self, state, epsilon):
    if epsilon > np.random.uniform(0, 1):#ランダム行動
      next_action = np.random.choice([0, 3])
    else:#Q値に従った行動
      a = np.where(self._Qtable[:,state[0],state[1]]==self._Qtable[:,state[0],state[1]].max())[0]
      next_action = np.random.choice(a)
    return next_action
#Q値の更新
  def update_Qtable(self, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.5
    next_maxQ=max(self._Qtable[:,next_state[0],next_state[1]])
    self._Qtable[action, state[0], state[1]] = (1 - alpha) * self._Qtable[action, state[0], state[1]] + alpha * (reward + gamma * next_maxQ)
    return self._Qtable

def main():
  num_episodes = 1  #総エピソード回数
  max_number_of_steps =100 #各エピソードの行動数
  epsilon = np.linspace(start=0.0, stop=0.0, num=num_episodes)#徐々に最適行動のみをとる、ε-greedy法
  env = MyEnvironmentSimulator()
  tab = MyQTable()
  for episode in range(num_episodes):  #エピソード回数分繰り返す
    state = env.reset()
    episode_reward = 0
    for t in range(max_number_of_steps):  #各エピソードで行う行動数分繰り返す
      action = tab.get_action(state, epsilon[episode]) #行動の決定
      next_state, reward = env.step(action) #行動による状態変化
#      q_table = tab.update_Qtable(state, action, reward, next_state)#Q値の更新
      state = next_state
      if reward!=0:
        break
    print(f'Episode:{episode}, Step:{t}, Reward:{reward}')
#  np.savetxt('Qvalue.txt', q_table.reshape(4*7*7))

#移動できているかのチェック
  state = [1,1]
  maze = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
  for t in range(100): #試行数分繰り返す
    maze[state[0], state[1]]=3
    action = np.where(tab._Qtable[:,state[0],state[1]]==tab._Qtable[:,state[0],state[1]].max())[0]
    print(t+1, state, action)
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
  print(maze)
  
if __name__ == '__main__':
  main()
