'''
入力の中の1の数を数えるプログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
  # データの作成
  input_data = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=np.float32, )# 入力用データ
  label_data = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32)# ラベル (教師データ)
  train_data, train_label = input_data, label_data  # 訓練データ
  validation_data, validation_label = input_data, label_data  # 検証データ
  # ニューラルネットワークの登録
  model = keras.Sequential(
    [
      keras.layers.Dense(6, activation='relu'),
      keras.layers.Dense(6, activation='relu'),
      keras.layers.Dense(4, activation='softmax'),
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
    epochs=1000,
    batch_size=8,
    validation_data=(validation_data, validation_label),
  )
  model.save(os.path.join('result', 'outmodel'))  # モデルの保存

if __name__ == '__main__':
  main()
