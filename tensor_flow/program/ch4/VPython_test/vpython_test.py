'''
Vpythonのテスト用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import vpython as vs

scene = vs.canvas(title='VPython test', x=0, y=0, width=600, height=400, center=vs.vector(0,0,0), background=vs.vector(1,1,1),autoscale = False)#環境の設定
ball = vs.sphere(pos=vs.vector(-5,0,0), radius=0.5, color=vs.vector(0.8,0.8,0.8))#ボールの設定
wall = vs.box(pos=vs.vector(5,0,0), size=vs.vector(0.2,5,5), color=vs.vector(0.5,0.5,0.5)) #壁の設定

ball.velocity = vs.vector(5,0,0)#ボールの速度

for t in range(300):
    vs.rate(100)#描画（100fps）
    ball.pos = ball.pos + ball.velocity*0.01#ボールの位置の更新
