import gym
import numpy as np
import cv2
import random
import csv
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

image_in = cv2.imread("smile_test.png")           #import image           (form of every pixel  [255, 255, 255] or [0, 0, 0]

image_out = np.ones(image_in.shape, np.uint8) * 255 #declaration  output image
image_in_array = np.zeros(image_in.shape)                   #declaration input array to have the image in another form (easier to work with)     black->1   white->0
image_out_array = np.zeros(image_in.shape)                  #declaration output array to have the image in another form (easier to change the value of pixels)     black->1   white->0



class FooEnv(gym.Env):


  def __init__(self, env):
    self.env = env

    self.action_space = self.env.action_space

    self.observation_space = self.env.observation_space

    self.reward_range = self.env.reward_range

    self.metadata = self.env.metadata

  def step(self, action):           # actionを実行し、結果を返す
      for y_pixel in range(0, image_in.shape[1]):
          for x_pixel in range(0, image_in.shape[0]):
              if all(image_in[x_pixel, y_pixel]) == True:
                  image_in_array[
                      x_pixel, y_pixel] = 0  # transform image to array           1 means black   0 means white
              else:
                  image_in_array[x_pixel, y_pixel] = 1
      x_pixel = random.randrange(0, image_in_array.shape[0], 1)  # last 1 is step1
      print("x = " + str(x_pixel))
      y_pixel = random.randrange(0, image_in_array.shape[1], 1)
      print("y = " + str(y_pixel))
      print("\n")
      plt.plot(x_pixel, y_pixel, color="red", marker="o", linestyle="None")
      which_action = random.randrange(1, 5, 1)  # 1 straight, 2 right, 3 back, 4 left
      action_position = []
      for i in range(0, 1000):
          action_position += str(which_action)
          with open('action3.csv', 'w') as file:
              writer = csv.writer(file, lineterminator='\n')
              writer.writerows(action_position)
          print(which_action)
          matrix_3x3 = [
              [image_in_array[x_pixel - 1][y_pixel - 1], image_in_array[x_pixel][y_pixel - 1], image_in_array[x_pixel + 1][y_pixel - 1]],
              [image_in_array[x_pixel - 1][y_pixel], image_in_array[x_pixel][y_pixel], image_in_array[x_pixel + 1][y_pixel]],
              [image_in_array[x_pixel - 1][y_pixel + 1], image_in_array[x_pixel][y_pixel + 1], image_in_array[x_pixel + 1][y_pixel + 1]]
                        ]
          image_out_array[x_pixel][y_pixel] = 1
          x1 = np.argmax(matrix_3x3)
          if x1 == 1:
              which_action = 1

          elif x1 == 5:
              which_action = 2

          elif x1 == 7:
              which_action = 3

          elif x1 == 3:
              which_action = 4

          else:
              which_action = random.randrange(1, 5, 1)

          if which_action == 1:
              y_pixel = y_pixel - 1

          elif which_action == 2:
              x_pixel = x_pixel + 1

          elif which_action == 3:
              y_pixel = y_pixel + 1

          elif which_action == 4:
              x_pixel = x_pixel - 1

      position = []
      for y_pixel in range(0, image_out_array.shape[1]):  # transform array to image       1->[0, 0, 0]    0->[255, 255, 255]
          for x_pixel in range(0, image_out_array.shape[0]):
              if all(image_out_array[x_pixel, y_pixel]) == True:  # if [x_pixel]==0 or [y_pixel]==0 -> false
                  image_out[x_pixel, y_pixel] = [0, 0, 0]
                  position += [[x_pixel, y_pixel]]
                  # print(position[-1])
                  with open('xy3.csv', 'w') as file:
                      writer = csv.writer(file, lineterminator='\n')
                      writer.writerows(position)
              else:
                  image_out[x_pixel, y_pixel] = [255, 255, 255]

      cv2.imshow("output image", image_out)  # displaying out image
      position_plot = []
      idx = position.index(position[-1])
      print(idx)
      a = int((idx + 1) / 10)
      for i in range(1, a):
          position_plot += [position[i * 10 - 1]]
      print(position_plot)
      plt.plot(position_plot, color="black", marker="o", linestyle="None")
      plt.show()
      return self.env.step(action)



  def reset(self):                  # 状態を初期化し、初期の観測値を返す


      return self.env.reset

  def render(self, close=False): # 環境を可視化する
      return self.env.render

  def close(self): # 環境を閉じて後処理をする


      return self.env.close()

  def seed(self, seed=None): # ランダムシードを固定する
      return self.env.seed(seed)