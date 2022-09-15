'''
カメラデータ収集用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import cv2
import os
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)#ピン配置の番号を使用
GPIO.setup(15, GPIO.OUT)#15番ピンを出力

def main():
  n_on, n_off = 0, 0
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()#画像の読み込み
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
    cv2.imshow('gray', gray)#画像表示
    key = cv2.waitKey(10)#キー入力
    if key == 97:#aキーの場合
      GPIO.output(15, 1)
      for _ in range(4):
        ret, frame = cap.read()#画像の読み込み
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
      cv2.imwrite(os.path.join('img',f'ON_{n_on}.png'), gray)#ONフォルダに保存
      print(f'ON: {n_on}')
      n_on = n_on + 1 
    elif key == 115:#sキーの場合
      GPIO.output(15, 0)
      for _ in range(4):
        ret, frame = cap.read()#画像の読み込み
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
      cv2.imwrite(os.path.join('img',f'OFF_{n_off}.png'), gray)#OFFフォルダに保存
      print(f'OFF: {n_off}')
      n_off = n_off + 1 
    elif key == 113:#qキーの場合
      break#ループを抜けて終了
  cap.release()

if __name__ == '__main__':
  main()
