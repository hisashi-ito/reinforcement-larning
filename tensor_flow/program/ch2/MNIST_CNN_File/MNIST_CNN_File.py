'''
ファイルから入力画像を読み込んで数字分類のためのCNN学習プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image # 追加

def main():
  # ファイルからの画像の読み込み
  img = Image.open(os.path.join('number', '2.png'))
  img = img.convert('L') # グレースケール変換
  img = img.resize((8, 8)) # 8×8にリサイズ
  img = 16.0 - np.asarray(img, dtype=np.float32) / 16.0 # 白黒反転，0-16に正規化，array化
  test_data = img.reshape(1, 8, 8, 1) # 4次元行列に変換（1×1×8×8，バッチ数×チャンネル数×縦×横）
  # ニューラルネットワークの登録
  model = keras.models.load_model(os.path.join('result', 'outmodel'))
  # 学習結果の評価
  prediction = model.predict(test_data)
  result = np.argmax(prediction)
  print(f'result: {result}')

if __name__ == '__main__':
  main()
