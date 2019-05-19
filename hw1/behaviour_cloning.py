import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.FileWriter(logdir + "/metrics")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('expert_policy_file', type=str)
# parser.add_argument('envname', type=str)
# parser.add_argument('--render', action='store_true')
# parser.add_argument("--max_timesteps", type=int)
# parser.add_argument('--num_rollouts', type=int, default=20,
#                     help='Number of expert roll outs')
# args = parser.parse_args()

# print('loading and building expert policy')
# policy_fn = load_policy.load_policy(args.expert_policy_file)
# print('loaded and built')

###########################################
# DEFINE MODEL
###########################################
class model_bc(tf.keras.Model):

  def __init__(self, num_classes):
    super(model_bc, self).__init__(name='model_bc')
    self.num_classes = num_classes
    self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
    self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
    self.dense_3 = tf.keras.layers.Dense(512, activation='relu')
    self.dense_4 = tf.keras.layers.Dense(num_classes)
    self.dropout = tf.keras.layers.Dropout(0.2)

  def call(self, inputs, training=False):
    x = self.dense_1(inputs)
    if training:
      x = self.dropout(x)
    x = self.dense_2(x)
    if training:
      x = self.dropout(x)
    x = self.dense_3(x)
    if training:
      x = self.dropout(x)
    return self.dense_4(x)

###########################################
#
###########################################
rewards_mean = []
def run_episodes():
  env = gym.make("HalfCheetah-v2")
  max_steps = env.spec.timestep_limit

  returns = []
  observations = []
  actions = []
  for i in range(5):
      print('iter', i)
      obs = env.reset()
      obs = (obs - obs_mean) / obs_std
      done = False
      totalr = 0.
      steps = 0
      while not done:
          action = model.predict(np.array(obs[None,:]))
          observations.append(obs)
          actions.append(action)
          obs, r, done, _ = env.step(action)
          obs = (obs - obs_mean) / obs_std
          totalr += r
          steps += 1
          
          # env.render()
          if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
          if steps >= max_steps:
              break
      returns.append(totalr)

  print('returns', returns)
  print('mean return', np.mean(returns))
  print('std of return', np.std(returns))
  print(env.action_space)
  print(env.observation_space)
  expert_data = {'observations': np.array(observations),
                  'actions': np.array(actions)}
  rewards_mean.append(np.mean(returns))
###########################################

###########################################

filename = "expert_data/HalfCheetah-v2_20.pkl"

with open(filename, 'rb') as f:
    data = pickle.loads(f.read())

obs, actions = np.squeeze(data['observations']), np.squeeze(data['actions'])
obs_mean = np.mean(obs, axis=0)
obs_std = np.std(obs, axis=0)
obs = (obs - obs_mean) / obs_std

x, x_test, y, y_test = train_test_split(obs, actions, test_size=0.2, train_size=0.8, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, train_size =0.75, random_state=0)

###########################################

###########################################

model = model_bc(y.shape[1])
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error


class rewards_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 1:
          run_episodes()

rewards_callback = rewards_callback()
###########################################

###########################################

model.fit(x_train, y_train, 
          validation_data = (x_val, y_val), 
          batch_size=64, 
          epochs=10, 
          verbose=1,
          callbacks=[tensorboard_callback, rewards_callback])

###########################################

###########################################


###########################################

###########################################

plt.plot(rewards_mean)
plt.show()


