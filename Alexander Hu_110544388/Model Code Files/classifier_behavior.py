import gym

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

###################
# Data Collection #
###################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integration")
)

dir = os.fsencode('./Test_Runs')

unique_keys = set()
data = []

for file in os.listdir(dir):
    fname = os.fsdecode(file)

    print("Now running: " + fname)

    movie = retro.Movie('./Test_Runs/'+fname)
    movie.step()

    env = retro.make("Excitebike-NES-Track-1",
        inttype=retro.data.Integrations.ALL,
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )

    env.initial_state = movie.get_state()
    env = ProcessExcitebikeFrame(env)
    cur_state = env.reset()

    #print("Made it past env")

    bad_result = 0
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        #print(keys)
        keys = tuple(keys)

        new_state, rew, done, info = env.step(keys)

        if keys != (False, False, False, False, False, False, False, False, False):
            unique_keys.add(keys)
            data.append((cur_state,keys))
        else:
            bad_result += 1

        cur_state = new_state
        if done:
            print("Finished race")
            print(bad_result)
            break

    env.close()

#################
# Preprocessing #
#################

unique_keys = list(unique_keys)
unique_keys.sort()
#print(unique_keys)
print(len(unique_keys))
print(len(data))

for k in unique_keys:
    print(k)

count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(data)):
    obs,act = data[i]
    a = unique_keys.index(act)
    count[a] += 1
    #array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #array[a] = 1
    data[i] = (obs,a)

print(count)
print(data[0][1])

storage_keys = []
for uk in unique_keys:
    arr = []
    for b in uk:
        arr.append(int(b))
    storage_keys.append(arr)

'''
for ska in storage_keys:
    print(ska)
'''

#store storage_keys in a document for referral
with open('storage_keys.txt', 'w') as f:
    for item in storage_keys:
        f.write(str(item) + '\n')
    f.close()

#split data into train and test
trainset,testset = sk.model_selection.train_test_split(data, train_size=0.7, test_size=0.3)
print(len(trainset))
print(len(testset))

####################
# Classifier Setup #
####################

env = retro.make("Excitebike-NES-Track-1",
    inttype=retro.data.Integrations.ALL
)
env = ProcessExcitebikeFrame(env)
observation_shapes = env.observation_space.shape
env.close()


lr = 0.0001

criterion = nn.CrossEntropyLoss()
behavior_model = CNN_Action_Model(observation_shapes, len(unique_keys)).to(device)
optimizer = optim.Adam(behavior_model.parameters(), lr=lr)

loss_array = []
count = 0
for epoch in range(2):
    running_loss = 0.0
    random.shuffle(trainset)
    for obs, act in trainset:

        optimizer.zero_grad()
        state_numpy_array = np.array([obs], copy=False)
        state_tensor = torch.tensor(state_numpy_array).float().to(device)
        outputs = behavior_model(state_tensor)
        act_tensor = torch.tensor(act).long().to(device)
        #print(outputs)
        #print(act_tensor.shape)
        loss = criterion(outputs,act_tensor.unsqueeze(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (count+1) % 1000 == 0:    # print every 2000 mini-batches
            loss_array.append(running_loss)
            print("Current count: " + str(count) + ", loss: " + str(running_loss))
            running_loss = 0.0
        count += 1

with open('classifier_loss_array.txt', 'w') as f:
    f.write(str(loss_array) + '\n')
    f.close()

torch.save(behavior_model.state_dict(), './classifier_model_excitebike.pth')

actual_results = []
pred_results = []
with torch.no_grad():
    for obs, true_act in testset:
        state_numpy_array = np.array([obs], copy=False)
        state_tensor = torch.tensor(state_numpy_array).float().to(device)
        pred_act_arr = behavior_model(state_tensor)
        pred_act = torch.argmax(pred_act_arr).detach().item()
        actual_results.append(true_act)
        pred_results.append(pred_act)

print(actual_results[:10])
print(pred_results[:10])

print(sk.metrics.classification_report(actual_results,pred_results))
