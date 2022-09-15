import tensorflow as tf

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import numpy as np
import os

#シミュレータクラスの設定
class EnvironmentSimulator(py_environment.PyEnvironment):
  def __init__(self):
    super(EnvironmentSimulator, self).__init__()
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, maximum=1
    )
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1
    )
    self._reset()
  def observation_spec(self):
    return self._observation_spec
  def action_spec(self):
    return self._action_spec
#初期化
  def _reset(self):
    self._state = 0
    return ts.restart(np.array([self._state], dtype=np.int32))
#行動による状態変化
  def _step(self, action):
    reward = 0
    if self._state == 0:#電源OFFの状態
      if action == 0:#電源ボタンを押す
        self._state = 1#電源ON
      else:#行動ボタンを押す
        self._state = 0#電源OFF
    else:#電源ONの状態
      if action == 0:
        self._state = 0
      else:
        self._state = 1
        reward = 1#報酬が得られる
    return ts.transition(np.array([self._state], dtype=np.int32), reward=reward, discount=1)#TF-Agents用の戻り値の生成

def main():
#環境の設定
  env_py = EnvironmentSimulator()
  env = tf_py_environment.TFPyEnvironment(env_py)
#行動の設定
  policy = tf.compat.v2.saved_model.load(os.path.join('policy'))

  episode_rewards = 0#報酬の計算用
  time_step = env.reset()#環境の初期化
  for t in range(5):#5回の行動
    policy_step = policy.action(time_step)#状態から行動の決定
    next_time_step = env.step(policy_step.action)#行動による状態の遷移

    S = time_step.observation.numpy().tolist()[0]
    A = policy_step.action.numpy().tolist()[0]
    R = next_time_step.reward.numpy().astype('int').tolist()[0]
    print(S, A, R)
    episode_rewards += R#報酬の合計値の計算

    time_step = next_time_step#次の状態を今の状態に設定

  print(f'Rewards:{episode_rewards}')
  env.close()

if __name__ == '__main__':
  main()
