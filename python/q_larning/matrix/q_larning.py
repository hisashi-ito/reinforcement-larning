#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【q_larning】
#
# 概要:
#      強化学習(Q 学習)の例題プログラム
#
# usage: pytho3 q_larning.py
#
# 更新履歴:
#          2020.01.08 新規作成
#          2020.01.08 Q 値の収束の具合を図示
#          2020.01.09 探索経路がマトリクス
#          2020.01.15
#
import os
import sys
import logging
import random
import math
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 定数定義
GEN_MAX = 50  # 学習の繰り返し回数
STATE_NO = 64 # 状態(8x8 のstate) の数
ACTION_NO = 4 # 行動の自由度(数)
ALPHA = 0.1   # 学習係数
GAMMA = 0.9   # 割引率
EPSILON = 0.3 # 行動選択のランダム率
REWARD = 100  # ゴール時の報酬
GOAL = 54     # ゴールの状態の番号
UP = 0        # 上方向への行動
DOWN = 1      # 下方向への行動
LEFT = 2      # 左方向へ移動
RIGHT = 3     # 右方向へ移動
LEVEL = 512   # 枝分かれの深さ

class QLarning(object):
    def __init__(self, logger, gen_max=GEN_MAX, state_no=STATE_NO,
                 action_no=ACTION_NO, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON, reward=REWARD,
                 goal=GOAL, up=UP, down=DOWN, left=LEFT, right=RIGHT, level=LEVEL):
        self.logger = logger
        # 設定値
        self.gen_max = gen_max
        self.state_no = int(state_no)
        self.action_no = int(action_no)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward = reward
        self.goal = goal
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.level = level
        
        # Q の値を乱数で初期化(境界条件を考慮)
        # 2次元配列の形状は [64,4]
        self.q = self._init_q()
        
        # Q 値の履歴情報(グラフを作成するために必要)
        self.q_hist = []

    def _init_q(self):
        """Q の初期化(境界条件含む)"""
        q = [[self._frand() for i in range(self.action_no)] for j in range(self.state_no)]
        for i in range(self.state_no):
            for _ in range(self.action_no):
                # 上辺の境界条件
                if i <= 7:
                    q[i][self.up] = -100.0
                # 下辺の境界条件
                if i >= 56:
                    q[i][self.down] = -100.0
                # 左辺の境界条件
                if i % 8 == 0:
                    q[i][self.left] = -100.0
                # 右辺の境界条件
                if i % 8 == 7:
                    q[i][self.right] = -100.0
        return q
        
    def _rand0or1(self):
        """0か1の数値の何方かを返却する乱数"""
        return random.randint(0, 1)

    def _rand03(self):
        """0から3の数値を返却する乱数"""
        return random.randint(0, 3)
    
    def _frand(self):
        """0～1の一様乱数"""
        return random.random()
    
    def _select_action(self, state):
        """行動の選択"""
        action = None
        # epsilon-greedy 法による行動選択
        if self._frand() < self.epsilon:
            # ランダムに行動を選択
            while True:
                action = self._rand03()
                # 境界条件でQ 値が正値であれば処理を終了
                if self.q[state][action] > 0.0:
                    return action
        else:
            return self._set_action_by_q(state)
        
    def _set_action_by_q(self, state):
        """状態が指定された場合にQ が大きいactionを選択する"""
        max_q = 0.0
        max_action = 0
        for action in range(self.action_no):
            # UP,DOWN,LEFT,RIGHT の状態を探索
            if self.q[state][action] > max_q:
                # 各状態で最大のQを持つaction を保存
                max_q = self.q[state][action]
                max_action = action
        return max_action
    
    def _nexts(self, status, action):
        """行動(action)のよって状態(status)をupdateする"""
        # 8x8 のマトリクスなので以下の構造となる
        next_state_matrix = [-8, 8, -1, 1]
        ret = status + next_state_matrix[action]
        return ret
    
    def _update_q(self, status, status_next, action):
        """Q の値を更新する"""
        qv = None
        if status_next == self.goal:
            # ゴールに到達した場合は報酬が付与される
            qv = self.q[status][action] + self.alpha * (self.reward - self.q[status][action])
        else:
            # 報酬がない場合
            qv = self.q[status][action] + self.alpha * (self.gamma * self.q[status_next][self._set_action_by_q(status_next)] - self.q[status][action])
        return qv
            
    def print_q_value(self):
        """Q の値を出力するユーティリティメソッド"""
        s = 0
        for state in self.q:
            x = "s:{} ↑: {} ↓: {} ←: {} →: {}".format(s, state[self.up], state[self.down], state[self.left], state[self.right])
            print(x)
            s += 1

    def print_max_q_value(self):
        arrows = ["↑", "↓", "←", "→"]
        # 8x8の行列で最大の数を表示する
        length = int(math.sqrt(self.state_no))
        pic = ""
        for i in range(length):
            _pic = ""
            for j in range(length):
                best_action = np.argmax(self.q[i+j])
                _pic += arrows[best_action]
            pic += _pic + "\n"
        print("***")
        print(pic)
        print("***")
        
    def graph(self, image):
        """Q 値の収束の具合を図示するための関数"""
        fig, ax = plt.subplots(ncols=1, figsize=(9,8))       
        x = np.asarray([ i for i in range(self.gen_max)])
        # self.q_hist.shape = [step, state, action]
        # q_hist.shape = [state, action, step]
        q_hist = np.asarray(self.q_hist).transpose(1,2,0)
        s = 0
        for up, down, left, right in q_hist:
            ax.plot(x, up, label="s:{},up".format(s))
            ax.plot(x, down, label="s:{},down".format(s))
            ax.plot(x, left, label="s:{},left".format(s))
            ax.plot(x, right, label="s:{},right".format(s))
            s += 1

        ax.set_title('Q-Larninig')
        ax.set_xlabel('generation')
        ax.set_ylabel('q value')
        ax.legend(loc='upper left')        
        fig.savefig(image)
        plt.close()
        
    def train(self):
        """学習関数"""
        # 世代のループ
        for gen in range(self.gen_max):
            # 行動の初期状態
            state = 0
            # 探索レベルのループ
            for lev in range(self.level):
                # 行動の選択
                action = self._select_action(state)
                #self.logger.info("status: {} action: {}".format(state, action))
                state_next = self._nexts(state, action)

                # Q の更新
                self.q[state][action] = self._update_q(state, state_next, action)
                
                # 行動により次の状態へ遷移
                state = state_next
                
                # ゴールしたら探索終了
                if state == self.goal:
                    break

            
            # Q の出力
            #self.print_q_value()
            self.print_max_q_value()
            # Q の履歴を保存する(グラフ用)
            self.q_hist.append(copy.deepcopy(self.q))
            

def main():
    # logger の設定
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s -- : %(message)s')
    logger = logging.getLogger()
    logger.info("*** start q_larning ***")
    ql = QLarning(logger)
    ql.train()
    ql.graph("q_value.png")
    logger.info("*** stop q_larning ***")

    
if __name__ == '__main__':
    main()
