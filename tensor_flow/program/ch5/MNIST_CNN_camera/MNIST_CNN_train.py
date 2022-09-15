'''
MNISTを用いた数字分類のためのCNN学習プログラム（画像の正規化）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def main():
  # データの作成
  digits = load_digits()
  train_data, valid_data, train_label, valid_label = train_test_split(digits.data,digits.target, test_size=0.2)
  train_data, valid_data = train_data/16.0, valid_data/16.0
  train_data = train_data.reshape((len(train_data), 8, 8, 1))
  valid_data = valid_data.reshape((len(valid_data), 8, 8, 1))
  # ニューラルネットワークの登録
  model = keras.Sequential(
    [
      keras.layers.Conv2D(16, 3, padding='same', activation='relu'),#畳み込み
      keras.layers.MaxPool2D(pool_size=(2, 2)),#プーリング
      keras.layers.Conv2D(64, 3, padding='same', activation='relu'),#畳み込み
      keras.layers.MaxPool2D(pool_size=(2, 2)),#プーリング
      keras.layers.Flatten(),#平坦化
      keras.layers.Dense(10, activation='softmax'),#全結合層
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
    validation_data=(valid_data, valid_label),
  )
  model.save(os.path.join('result', 'outmodel'))  # モデルの保存

if __name__ == '__main__':
  main()
