#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【q_larning】
#
# 概要:
#      強化学習(Q 学習)の例題プログラム
#
import logging
import random

# 定数定義
GEN_MAX = 50  # 学習の繰り返し回数
STATE_NO = 7  # 状態(state) の数
ACTION_NO = 2 # 行動の自由度(数)
ALPHA = 0.1   # 学習係数
GAMMA = 0.9   # 割引率
EPSILON = 0.3 # 行動選択のランダム率
REWARD = 10   # ゴール時の報酬
GOAL = 6      # ゴールの状態の番号
UP = 0        # 上方向への行動
DOWN = 1      # 下方向への行動
LEVEL = 2     # 枝分かれの深さ

class QLarning(object):
    def __init__(self, logger, gen_max=GEN_MAX, state_no=STATE_NO,
                 action_no=ACTION_NO, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON, reward=REWARD,
                 goal=GOAL, up=UP, down=DOWN, level=LEVEL):
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
        self.level = level

        # Q の値を乱数で初期化
        # 2次元配列の形状は [7,2]
        self.q = [[self._frand() for i in range(self.action_no)] for j in range(self.state_no)]
        
    def _rand0or1(self):
        """0か1の数値の何方かを返却する乱数"""
        return random.randint(0, 1)

    def _frand(self):
        """0～1の一様乱数"""
        return random.random()
    
    def _select_action(self, state):
        """行動の選択"""
        action = None
        # epsilon-greedy 法による行動選択
        if self._frand() < self.epsilon:
            # ランダムに行動を選択
            action = self._rand0or1()
        else:
            action = self._set_action_by_q(state)
        return action
        
    def _set_action_by_q(self, state):
        """状態が指定された場合にQが大きいactionを選択する"""
        if self.q[state][self.up] > self.q[state][self.down]:
            return self.up
        else:
            return self.down

    def _nexts(self, status, action):
        """行動(action)のよって状態(status)をupdateする"""
        # この式は今の設定(bi-treeの状態じゃないといけない)
        return 2 * status + 1 + action

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
            x = "s:{} ↑: {} ↓:{}".format(s, state[self.up], state[self.down])
            print(x)
            s += 1
            
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
                self.logger.info("status: {} action: {}".format(state, action))
                state_next = self._nexts(state, action)

                # Q の更新
                self.q[state][action] = self._update_q(state, state_next, action)

                # 行動により次の状態へ遷移
                state =state_next

            # Q の出力
            self.print_q_value()

            
def main():
    # logger の設定
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s -- : %(message)s')
    logger = logging.getLogger()
    logger.info("*** start q_larning ***")
    QLarning(logger).train()
    logger.info("*** stop q_larning ***")

    
if __name__ == '__main__':
    main()
