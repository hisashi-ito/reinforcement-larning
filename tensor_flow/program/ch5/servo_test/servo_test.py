'''
RaspberryPi用
RCサーボモータテスト用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)#サーボモータの周期の設定
while True:
  angle = input('[200-600]:')#200から600までの数値を入力
  pwm.set_pwm(0,0,int(angle))#ドライバの接続位置，i2cdetectで調べた番号，サーボモータの回転
