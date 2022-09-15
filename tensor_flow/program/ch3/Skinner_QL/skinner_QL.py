'''
ネズミ学習問題のQラーニングプログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import numpy as np
#シミュレータクラスの設定
class MyEnvironmentSimulator():
  def __init__(self):
    self.reset()
#初期化
  def reset(self):
    self._state = 0
    return self._state
#行動による状態変化
  def step(self, action):
    reward = 0
    if self._state==0:#電源OFFの状態
      if action==0:#電源ボタンを押す
        self._state = 1#電源ON
      else:#行動ボタンを押す
        self._state = 0#電源OFF
    else:#電源ONの状態
      if action==0:
        self._state = 0
      else:
        self._state = 1
        reward = 1#報酬が得られる
    return self._state, reward
#Q値クラスの設定
class MyQTable():
  def __init__(self):
    self._Qtable = np.zeros((2, 2))
#行動の選択
  def get_action(self, state, epsilon):
    if epsilon > np.random.uniform(0, 1):#ランダム行動
      next_action = np.random.choice([0, 1])
    else:#Q値に従った行動
      a = np.where(self._Qtable[state]==self._Qtable[state].max())[0]
      next_action = np.random.choice(a)
    return next_action
#Q値の更新
  def update_Qtable(self, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.5
    next_maxQ=max(self._Qtable[next_state])
    self._Qtable[state, action] = (1 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
    return self._Qtable

def main():
  num_episodes = 10  #総エピソード回数
  max_number_of_steps =5 #各エピソードの行動数
  epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes)#徐々に最適行動のみをとる、ε-greedy法
  env = MyEnvironmentSimulator()
  tab = MyQTable()

  for episode in range(num_episodes):  #エピソード回数分繰り返す
    state = env.reset()
    episode_reward = 0
    for t in range(max_number_of_steps):  #各エピソードで行う行動数分繰り返す
      action = tab.get_action(state, epsilon[episode]) #行動の決定
      next_state, reward = env.step(action) #行動による状態変化
      print(state, action, reward)#表示
      q_table = tab.update_Qtable(state, action, reward, next_state)#Q値の更新
      state = next_state
      episode_reward += reward  #報酬を追加
    print(f'Episode:{episode+1:4.0f}, Reward:{episode_reward:3.0f}')
    print(q_table)
  np.savetxt('Qvalue.txt', tab._Qtable)
  
if __name__ == '__main__':
  main()
