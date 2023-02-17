import gym
from gym.spaces import MultiBinary

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

import math
import glob
import io
import base64
import time
import pdb
import cv2
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import retro

import sklearn as sk
import sklearn.model_selection
import sklearn.metrics

from collections import namedtuple, deque
import collections
from itertools import count

# Colab comes with PyTorch
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

###################
# Neural Network #
##################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Action_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim[1] != 90:
            raise ValueError(f"Expecting input height: 90, got: {input_dim[1]}")
        if input_dim[2] != 90:
            raise ValueError(f"Expecting input width: 90, got: {input_dim[2]}")

        #print("CNN_Action_Model inputs/outputs")
        #print(input_dim)
        #print(output_dim)

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(3136, 6000),
            nn.ReLU(),
            nn.Linear(6000,3000),
            nn.ReLU(),
            nn.Linear(3000,1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, input):
        #print(input)
        feature_v = self.net1(input)
        #print("Feature vector shape:")
        #print(feature_v.shape)
        return self.net2(feature_v)

################################
# Wrappers and Other Functions #
################################

class ProcessExcitebikeFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessExcitebikeFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 90, 90), dtype=np.uint8)

    def observation(self, obs):
        return ProcessExcitebikeFrame.process(obs)

    @staticmethod
    def process(frame):
        #check if right dimensions
        if frame.size == 224 * 240 * 3:
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        #crop out the top and left portions as they are unnecessary
        #crop out bit of the right as it shouldn't make too much of an impact on training
        crop_obs = img[31:211,45:225]

        #colored out the time on bottom to black
        crop_obs[160:180,0:56] = np.array([0,0,0],dtype=np.float32)
        crop_obs[160:180,95:] = np.array([0,0,0],dtype=np.float32)

        #convert to greyscale
        crop_obs = crop_obs[:, :, 0] * 0.299 + crop_obs[:, :, 1] * 0.587 + crop_obs[:, :, 2] * 0.114

        #resize from 180x180 to 90x90, also eliminating the 1 channel at the end
        resized = cv2.resize(crop_obs, (90,90),interpolation = cv2.INTER_AREA)
        resized = np.reshape(resized, [90, 90*1])

        #add a dimension in the front for the sake of nn processing
        resized = np.expand_dims(resized, axis=0)
        return resized.astype(np.uint8)



# Unzip iterable function definition
def unzip(iterable):
    return zip(*iterable)

#################
# Loading Data #
################

unique_keys = []

f = open("storage_keys.txt", "r")
for k in f.readlines():
    k_arr = k.split(",")
    #print(k_arr)
    k_action = []
    for i in range(len(k_arr)):
        key = int(k_arr[i][1])
        k_action.append(key)
    #print(k_action)
    unique_keys.append(k_action)
f.close()


#####################
# Environment Setup #
#####################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integration")
)

env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL)
env = ProcessExcitebikeFrame(env)
#print(env)
observation_shapes = env.observation_space.shape

behavior_model = CNN_Action_Model(observation_shapes, len(unique_keys)).to(device)
state_dict = torch.load('./classifier_model_excitebike.pth')
behavior_model.load_state_dict(state_dict)

num_iter = 10

obs = env.reset()
'''
results = []
for i in range(num_iter):
    total_rew = 0
    while True:
        state_numpy_array = np.array([obs], copy=False)
        state_tensor = torch.tensor(state_numpy_array).float().to(device)

        pred_act_arr = behavior_model(state_tensor)
        pred_act = torch.argmax(pred_act_arr).detach().item()
        act = unique_keys[pred_act]

        new_obs, rew, done, info = env.step(act)
        if rew != 5000:
            rew = -1
        total_rew += rew

        obs = new_obs
        #print(act)
        #env.render()
        #print(rew)

        if done:
            print("Iteration " + str(i) + " - Reward: " + str(total_rew))
            results.append(total_rew)
            obs = env.reset()
            break
'''
env.close()


env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL)
env = ProcessExcitebikeFrame(env)
obs = env.reset()

#test_act = env.action_space.sample()
#print(type(test_act))
#print(test_act.shape)
#print(test_act)

total_rew = 0
moves = []
#print("Before first loop")
while True:
    state_numpy_array = np.array([obs], copy=False)
    state_tensor = torch.tensor(state_numpy_array).float().to(device)

    pred_act_arr = behavior_model(state_tensor)
    pred_act = torch.argmax(pred_act_arr).detach().item()
    act = unique_keys[pred_act]
    act = np.array(act)
    #print(act)
    #print(type(act))
    #print(act.shape)
    #act = tuple(act)

    #test_mb = MultiBinary(9)
    #print(test_mb.shape)

    moves.append(act)

    new_obs, rew, done, info = env.step(act)
    if rew != 5000:
        rew = -1
    total_rew += rew

    obs = new_obs
    #print(act)
    #env.render()
    #print(rew)

    if done:
        print("Reward: " + str(total_rew))
        #results.append(total_rew)
        obs = env.reset()
        break
env.close()

#print("Excitebike-NES-Track-1" in retro.data.list_games(inttype=retro.data.Integrations.ALL))


def main(mv):
    env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL, record=".")
    obs = env.reset()

    total_rew = 0
    for a in mv:

        new_obs, rew, done, info = env.step(a)
        if rew != 5000:
            rew = -1
        total_rew += rew

        obs = new_obs

        if done:
            print("Reward: " + str(total_rew))

            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main(moves)
