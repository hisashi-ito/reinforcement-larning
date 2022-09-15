'''
カメラから入力画像を読み込んで数字分類のためのCNN学習プログラム（28x28画像使用）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

def main():
  # ニューラルネットワークの登録
  model = keras.models.load_model(os.path.join('result', 'outmodel'))

  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()#画像の読み込み
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
    xp = int(frame.shape[1]/2)
    yp = int(frame.shape[0]/2)
    d = 28*2
    cv2.rectangle(gray, (xp-d, yp-d), (xp+d, yp+d), color=0, thickness=2)
    cv2.imshow('gray', gray)#画像表示
    gray = cv2.resize(gray[yp-d:yp + d, xp-d:xp + d],(28, 28))
    img = np.zeros((28,28), dtype=np.float32)
    img[np.where(gray>64)]=1
    img = 1-np.asarray(img,dtype=np.float32)  # 反転処理
    test_data = img.reshape(1, 28, 28, 1) # 4次元行列に変換（1×8×8×1，バッチ数×縦×横×チャンネル数）
    prediction = model.predict(test_data)# 学習結果の評価
    result = np.argmax(prediction)
    print(f'result: {result}')
    key = cv2.waitKey(10)#キー入力
    if key == 113:#qキーの場合
      break#ループを抜けて終了
  cap.release()

if __name__ == '__main__':
  main()
