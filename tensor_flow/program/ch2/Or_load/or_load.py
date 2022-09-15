'''
論理演算子ORの学習プログラム（モデル入力）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
  # データの作成
  test_data = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=np.float32, )# 入力用データ
  # ニューラルネットワークの登録
  model = keras.models.load_model(os.path.join('result', 'outmodel'))
  # 学習結果の評価
  predictions = model.predict(test_data)
  print(predictions)
  for i, prediction in enumerate(predictions):
    result = np.argmax(prediction)
    print(f'input: {test_data[i]}, result: {result}')

if __name__ == '__main__':
  main()
