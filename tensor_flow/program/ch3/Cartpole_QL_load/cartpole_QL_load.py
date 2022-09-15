'''
倒立振子のQラーニングプログラム（再スタート）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import gym
import numpy as np
import time
#Q値クラスの設定
class MyQTable():
  def __init__(self, num_action):
    #self._Qtable = np.random.uniform(low=-1, high=1, size=(num_digitized**4, num_action))
    self._Qtable = np.loadtxt('Qvalue.txt')
#行動の選択
  def get_action(self, next_state, epsilon):
    if epsilon > np.random.uniform(0, 1):
      next_action = np.random.choice([0, 1])
    else:
      a = np.where(self._Qtable[next_state]==self._Qtable[next_state].max())[0]
      next_action = np.random.choice(a)
    return next_action
#Q値の更新
  def update_Qtable(self, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_maxQ=max(self._Qtable[next_state])
    self._Qtable[state, action] = (1 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
    return self._Qtable

num_digitized = 6  #分割数
def digitize_state(observation):
  p, v, a, w = observation
  d = num_digitized
  pn = np.digitize(p, np.linspace(-2.4, 2.4, d+1)[1:-1])
  vn = np.digitize(v, np.linspace(-3.0, 3.0, d+1)[1:-1])
  an = np.digitize(a, np.linspace(-0.5, 0.5, d+1)[1:-1])
  wn = np.digitize(w, np.linspace(-2.0, 2.0, d+1)[1:-1])
  return pn + vn*d + an*d**2 + wn*d**3

def main():
  num_episodes = 2000  #総試行回数
  max_number_of_steps = 200  #1試行のstep数
  env = gym.make('CartPole-v0')
  tab = MyQTable(env.action_space.n)
  for episode in range(1000,num_episodes+1):  #試行数分繰り返す
    observation = env.reset()
    state = digitize_state(observation)
    episode_reward = 0
    for t in range(max_number_of_steps):  #1試行のループ
      action = tab.get_action(state, epsilon = 0.5 * (1 / (episode + 1))) #行動の決定
      observation, reward, done, info = env.step(action) #行動による状態変化
      if episode %10 == 0:#表示
        env.render()
      if done and t < max_number_of_steps-1:
        reward -= max_number_of_steps  #棒が倒れたら罰則
      next_state = digitize_state(observation)  #t+1での観測状態を、離散値に変換
      q_table = tab.update_Qtable(state, action, reward, next_state)#Q値の更新
      state = next_state
      episode_reward += reward  #報酬を追加
      if done:
        break
    print(f'Episode:{episode:4.0f}, R:{episode_reward:4.0f}')
  np.savetxt('Qvalue.txt', tab._Qtable)

if __name__ == '__main__':
  main()
