#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【q_larning】
#
# 概要:
#      強化学習(Q 学習)の例題プログラム
#

# 定数定義
GEN_MAX = 50  # 学習の繰り返し回数
STATENO = 7   # 状態(state) の数
ACTIONNO = 2  # 行動の自由度(数)
ALPHA = 0.1   # 学習係数
GAMMA = 0.9   # 割引率
EPSILON = 0.3 # 行動選択のランダム率
SEED = 32767  # 乱数のシード
REWARD = 10   # ゴール時の報酬
GOAL = 6      # ゴールの状態の番号
UP = 0        # 上方向への行動
DOWN = 1      # 下方向への行動
LEVEL = 2     # 枝分かれの深さ

class QLarning(object):
    def __init__(self):
        pass

    
