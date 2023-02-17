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
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
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

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class EBikeDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[['A'], ['B'], ['A', 'UP'], ['A', 'DOWN'], ['A', 'RIGHT'], ['B','UP'], ['B','DOWN'], ['B','RIGHT']])

# Unzip iterable function definition
def unzip(iterable):
    return zip(*iterable)

#################
# Replay Buffer #
#################

SingleExperience = namedtuple('SingleExperience',
                              field_names = ['state','action','reward','done','nextstate'])

class ReplayBuffer:
    def __init__(self,size):
        self.buffer = deque(maxlen = size)
    def sampleBuf(self,size):
        # First let us get a list of elements to sample
        # Make sure to choose replace = False so that we cannot choose the same
        # sample tuples multiple times
        el_inds = np.random.choice(len(self.buffer), size, replace=False)

        # A nifty piece of code implemented by @Jordi Torres, this makes use of the
        # zip function to combine each respective data field, and returns the np arrays
        # of each separate entity.
        arr_chosen_samples = [self.buffer[i] for i in el_inds]
        # Take the samples and break them into their respective iterables
        state_arr, actions_arr, reward_arr, done_arr, next_state_arr = unzip(arr_chosen_samples)
        # Return these iteratables as np arrays of the correct types
        return np.array(state_arr),np.array(actions_arr),np.array(reward_arr,dtype=np.float32),np.array(done_arr, dtype=np.uint8),np.array(next_state_arr)

    def append(self, sample):
        self.buffer.append(sample)

    def size(self):
        return len(self.buffer)

#########
# Agent #
#########

class Agent:
    def __init__(self, buffer, model_buffer, environment):
        # Set object variables
        self.env = environment
        self.replay_buffer = buffer
        self.model_buffer = model_buffer
        self.restart_episode()

    # Restarts environment, and resets all necessary variables
    def restart_episode(self):
        self.state = self.env.reset()
        self.total_reward = float(0)

    # Define epsilon greedy policy
    def choose_epsilon_greedy_action(self,model,epsilon,device="cpu"):
        if random.uniform(0,1) > epsilon:
            state_numpy_array = np.array([self.state], copy=False)
            state_tensor = torch.tensor(state_numpy_array).float().to(device)
            model_estimated_action_values = model(state_tensor)
            #print("model estimate action value shape")
            #print(model_estimated_action_values.shape)
            act_v = torch.argmax(model_estimated_action_values, dim=1) # This is the same as torch.argmax
            action = int(act_v.item())
        else:
            action = env.action_space.sample()
        return action

    # Function that interacts one step with the environment, adds that sample
    # to the replay buffer, and returns the final cumulative reward or None if
    # episode hasn't terminated

    # We pass in the epsilon for choosing random actions
    # Let's also define the device for standard practice, and so that we can use
    # GPU training effectively.
    def advance_state_and_log(self, model, epsilon, device="cpu"):
        # First, let us choose an action to take using epsilon-greedy
        action = self.choose_epsilon_greedy_action(model,epsilon,device)
        # Now that we have chosen an action, take a step and increment total reward count
        observation, reward, done, info = self.env.step(action)
        if reward != 5000:
            reward = -1
        self.total_reward += reward

        # Now that we have the output, we must append a value to our replay buffer
        # First we must actually create a tuple according to the object we specified
        # Remember: field_names = ['state','action','reward','done','nextstate'])
        sample_tuple = SingleExperience(self.state, action, reward, done, observation)
        # Add to our buffer
        self.replay_buffer.append(sample_tuple)
        if done:
            self.model_buffer.append(sample_tuple)

        # Update our current state
        self.state = observation

        if done:
            t_reward = self.total_reward
            self.restart_episode()
            return (True, t_reward)
        return (False, None)

###################
# Algorithm Setup #
###################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integration")
)
env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL, record = ".")
env = EBikeDiscretizer(env)
env = ProcessExcitebikeFrame(env)
print('EBikeDiscretizer action_space', env.button_combos)
#print(env.unwrapped.get_action_meanings())

obs = env.reset()

# Define environment inputs and outputs
observation_shapes = env.observation_space.shape
print(observation_shapes)
num_actions = env.action_space.n
print(num_actions)

# Set up replay buffer parameters
buffer_size = 20000
batch_size = 128

# Define learning rate and other learning parameters
lr = 0.0001
gamma = 0.9
sync_target_frequency = 2000
max_steps = 1000000

# Epsilon parameters
epsilon = 1
decay = 0.99999
min_epsilon = 0.01

# Model learning parameters
model_size = 10

# Initialize networks
behavior_model = CNN_Action_Model(observation_shapes, num_actions).to(device)
state_dict = torch.load('./dqn2_model_excitebike.pth')
behavior_model.load_state_dict(state_dict)
behavior_model.eval()

# Define optimizer as Adam, with our learning rate
# optimizer = optim.Adam(behavior_model.parameters(), lr=lr)

# Initialize the buffer and agent
replay_buffer = ReplayBuffer(buffer_size)
model_buffer = ReplayBuffer(model_size) #use this to store done states
agent = Agent(replay_buffer, model_buffer, env)

#############
# Algorithm #
#############

# Let's log the returns, and step in episode
return_save = []
loss_save = []
episode_num = 0

def test_model_with_greedy_policy(env,model,agent,num_iterations):
    # Keep track of episodic-return
    test_episode_returns = np.zeros((num_iterations))

    agent.restart_episode()

    # For number of episodes
    for i in range(num_iterations):

        while True:
            #your agent goes here
            is_done, ret = agent.advance_state_and_log(model,0,device=device)

            if is_done:
              # Add return to save
              test_episode_returns[i] = ret
              break;

    return test_episode_returns


returns_save_dqn_atari = test_model_with_greedy_policy(env,behavior_model,agent,1)

print("Returns for 10 episodes with greedy final policy:")
print(returns_save_dqn_atari)
env.close()
