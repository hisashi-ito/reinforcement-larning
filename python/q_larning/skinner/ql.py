#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【ql】
#
import numpy as np


class Simulator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        reward = 0
        if self._state == 0:
            # 電源OFFの場合
            if action == 0:
                # 電源ボタンを押下されたので電源ON 状態
                self._state = 1
        else:
            # 電源ON の場合
            if action == 0:
                # 電源ボタンを押下されたので電源OFF 状態
                self._state = 0
            else:
                # 行動ボタンを押下されたのでリワード発生
                reward = 1

        return self._state, reward

class QTable(object):
    def __init__(self):
        self._Qtable = np.zeros((2, 2))

    def get_action(self, state, epsilon):
        if epsilon > np.random.uniform(0, 1):
            # epsilon の値が1.0 の場合はすべてQ 値に従った動作
            # epsilon の値が0.3 の場合は70% はQ 値に従った動作
            next_action = np.random.choice([0, 1])
        else:
            # Q 値に従った行動
            # 同じ状態の中で最大の行動を選択
            a = np.where(self._Qtable[state] == self._Qtable[state].max())[0]
            next_action = np.random.choice(a)

        return next_action

    def update_Qtable(self, state, action, reward, next_state):
        gamma = 0.9
        alpha = 0.5

        next_maxQ = max(self._Qtable[next_state])
        self._Qtable[state, action] = (1.0 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
        return self._Qtable


def main():
    # 総エピソード回数
    num_episodes = 10
    # 各エピソードの行動数
    max_number_of_steps = 5
    epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes)

    env = Simulator()
    tab = QTable()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_reward = 0

        # 各エピソードで行う行動数分繰り返す
        for t in range(max_number_of_steps):
            action =  tab.get_action(state, epsilon[episode])
            next_state, reward = env.step(action)
            print(state, action, reward)
            q_table = tab.update_Qtable(state, action, reward, next_state)
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode+1:4.0f}, R:{episode_reward:3.0f}")
        print(q_table)

    np.savetxt("Qvalue.txt", tab._Qtable)
        
if __name__ == '__main__':
    main()
    
