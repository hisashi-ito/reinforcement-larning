'''
MNISTを用いた数字分類のためのDNN学習プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.datasets import load_digits#データ用
from sklearn.model_selection import train_test_split#分割用

def main():
  # データの作成
  digits = load_digits()
  train_data, validation_data, train_label, validation_label = train_test_split(digits.data,digits.target, test_size=0.2)
  # ニューラルネットワークの登録
  model = keras.Sequential(
    [
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dense(10, activation='softmax'),
    ]
  )
  # 学習のためのmodelの設定
  model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
  )
  # 学習の実行
  model.fit(
    x = train_data,
    y = train_label,
    epochs=20,
    batch_size=100,
    validation_data=(validation_data, validation_label),
  )
  model.save(os.path.join('result', 'outmodel'))  # モデルの保存

if __name__ == '__main__':
  main()
