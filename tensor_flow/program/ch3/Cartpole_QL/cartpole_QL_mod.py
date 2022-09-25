#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import time

class QTable(object):
    def __init__(self, num_action, num_digitized=6):
        # 状態数: 6(状態数) ** 4(自由度) = 1296
        #
        # 状態数各自由度を6段階に離散化したもの(0,1,2,3,4,5)
        # 自由度は４つあり
        #
        # 1. カートの位置
        # 2. カートの速度
        # 3. ポールの角度
        # 4, ポーツの角速度
        # 
        # https://github.com/openai/gym/wiki/CartPole-v0
        #
        self._Qtable = np.random.uniform(low=-1, high=1, size=(num_digitized**4, num_action))

    def get_action(self, state, epsilon):
        if epsilon > np.random.uniform(0, 1):
            # `eposilon` が1に近くなるほどランダム
            next_action = np.random.choice([0, 1])
        else:
            # Qの最大値
            a = np.where(self._Qtable[state] == self._Qtable[state].max())[0]
            next_action = np.random.choice(a)
            # 同値の値がある場合はランダムでよい

        return next_action
        
    def update_Qtable(self, state, action, reward, next_state):
        # 割引率
        gamma = 0.99
        # 学習率
        alpha = 0.5
        next_maxQ = max(self._Qtable[next_state])

        # Q値の更新
        self._Qtable[state, action] = (1 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
        return self._Qtable[state, action]

def digitize_state(observation, num_digitized=6):
    # Open Gym が出してくる観測値
    # カートの位置, カートの速度, ポールの角度, ポールの角速度
    p, v, a, w = observation
    d = num_digitized

    # 連続値を離散化
    pn = np.digitize(p, np.linspace(-2.4, 2.4, d+1)[1:-1])
    vn = np.digitize(v, np.linspace(-3.0, 3.0, d+1)[1:-1])
    an = np.digitize(a, np.linspace(-0.5, 0.5, d+1)[1:-1])
    wn = np.digitize(w, np.linspace(-2.0, 2.0, d+1)[1:-1])

    # インデクスの変換する
    return pn + vn*d + an*d**2 + wn*d**3

def main():
    # 総エピソード数
    num_episodes = 1000
    # 各エピソードの最大行動数
    max_number_of_steps = 200
    # 離散化数
    num_digitized = 6
    
    env = gym.make('CartPole-v0')
    tab = QTable(env.action_space.n) # action数は2
    
    # エピソードのループ
    for episode in range(num_episodes+1):
        # 環境をリセットし観測値を得る
        observation = env.reset()

        # 観測値(連続値)から一意のindexに変換
        state = digitize_state(observation)
        
        episode_reward = 0

        # stepのloop
        for t in range(max_number_of_steps):
            # エピソードが進むにつれてepsilonが小さくなるようにしている
            # 最小でも0.5になるようにしている
            epsilon = 0.5 * (1.0 / (episode + 1.0))
            action = tab.get_action(state, epsilon=epsilon)
            
            # 行動による状態変化
            observation, reward, done, info = env.step(action) 

            if episode % 10 == 0:
                env.render()
                
            if done and t < max_number_of_steps - 1:
                # 棒が倒れたたら max_number_of_steps をマイナス
                reward = reward - max_number_of_steps

            # 観測値から対応する状態を取得インデクスを取得
            next_state = digitize_state(observation)

            # Q値の値を更新
            q_table = tab.update_Qtable(state, action, reward, next_state)
            state = next_state
            # 補修を追加
            episode_reward += reward
            if done:
                break

            print(f'Episode:{episode:4.0f}, Reward:{episode_reward:4.0f}')

        np.savetxt('Qvalue.txt', tab._Qtable)


if __name__ == '__main__':
    main()
