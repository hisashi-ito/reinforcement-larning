'''
Gym確認用プログラム(倒立振子)
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
  env.render()
  env.step(env.action_space.sample())
