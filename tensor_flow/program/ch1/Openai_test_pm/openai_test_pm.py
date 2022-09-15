'''
Gym[atari]確認用プログラム(パックマン)
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import gym
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
  env.render()
  env.step(env.action_space.sample())
