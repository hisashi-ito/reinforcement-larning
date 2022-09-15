'''
PyBulletのテスト用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import pybullet as p 
import time 

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8) #重力加速度の方向と大きさの設定
ballId = p.loadURDF("ball.urdf",[0,0,1], p.getQuaternionFromEuler([0,0,0]))#ボールの設定と位置角度の設定
p.resetBaseVelocity(ballId,[0,4,6])#ボールの速度の設定
p.changeDynamics(ballId, -1, linearDamping=0, angularDamping=0)#運動の設定
dt = 0.01#刻み時間
p.setTimeStep(dt)#刻み時間の設定
for i in range (150):
  pos, orn = p.getBasePositionAndOrientation(ballId) #ボールの位置と角度
  print(i*dt, '\t', pos[1], '\t', pos[2]) #コンソールに表示
  p.stepSimulation()#シミュレーションの実行と表示
  time.sleep(dt) #見やすくするための待ち時間
p.disconnect()
