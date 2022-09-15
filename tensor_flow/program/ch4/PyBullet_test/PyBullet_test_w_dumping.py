'''
PyBulletのテスト用プログラム（抵抗，摩擦あり）
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import pybullet as p 
import time 

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8) 
ballId = p.loadURDF("ball.urdf",[0,0,1], p.getQuaternionFromEuler([0,0,0]))
p.resetBaseVelocity(ballId,[0,4,6])
p.changeDynamics(ballId, -1)
dt = 0.01#1./240.
p.setTimeStep(dt)
for i in range (150):
  pos, orn = p.getBasePositionAndOrientation(ballId) 
  print(i*dt, '\t', pos[1], '\t', pos[2]) 
  p.stepSimulation()
  time.sleep(dt) 
p.disconnect()
