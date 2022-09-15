'''
RaspberryPi用
出力テスト用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)#ピン配置の番号を使用
GPIO.setup(15, GPIO.OUT)#22番ピンを入力

while True:
  angle = input('[a or s]:')#aもしくはsを入力
  if angle == 'a':
    GPIO.output(15, 1)
  elif angle == 's':
    GPIO.output(15, 0)
