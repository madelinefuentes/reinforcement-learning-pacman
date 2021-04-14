import retro
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import random
import time
import psutil

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.callbacks import History

# fill memory with experience
def pretrain(pretrain_length, memory_capacity, env, possible_actions):
    memory = ReplayMemory(memory_capacity)
    stacked_frames = []

    for i in range (pretrain_length):
        if i == 0:
            state = env.reset()
            stacked_frames = add_state(None, state, True)

        action = random.randint(0, len(possible_actions)-1)
        new_state, reward, done, info = env.step(possible_actions[action])
        new_stacked_frames = add_state(stacked_frames, new_state, False)
        memory.add((format_frames(stacked_frames), action, reward, 
            format_frames(new_stacked_frames), done))

        #env.render()
        if done:
            state = env.reset()
            stacked_frames = add_state(None, state, True)
        else:
            stacked_frames = new_stacked_frames
    env.close
    return memory

# format stacked frames for input to model
def format_frames(frames):
    np_frames = np.asarray(frames).astype('float32')
    return np_frames.reshape(125, 80, 4)

def train():
    # hyperparameters
    episodes = 50
    max_steps_per_episode = 2000
    state_space_size = [125, 80, 4]
    memory_capacity = 1000000

    learning_rate = 0.001 #alpha
    discount_rate = 0.9 #gamma
    batch_size = 16
    max_tau = 1000
    exploration_rate = 1
    max_exploration = 1
    min_exploration = 0.01
    exploration_decay_rate = 0.05

    env = retro.make(game = "PacManNamco-Nes")
    env.frameskip = 2
    possible_actions = [[0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0]]

    action_space_size = 4

    policy_model = DeepQNetwork(state_space_size, action_space_size, learning_rate)
    policy_model = policy_model.model
    target_model = DeepQNetwork(state_space_size, action_space_size, learning_rate)
    target_model = target_model.model
    target_model.set_weights(policy_model.get_weights()) 

    history = History()
    render_episode = True
    tau = 0

    memory = pretrain(70000, memory_capacity, env, possible_actions)

    for episode in range(episodes):
        state = env.reset()
        stacked_frames = add_state(None, state, True)
        episode_rewards = []

        for step in range(max_steps_per_episode):
            # exploration vs exploitation
            exploration_probability = random.uniform(0, 1)

            if(exploration_probability > exploration_rate):
                action_probability = policy_model.predict(format_frames(stacked_frames).reshape(1, 125, 80, 4))
                action = np.argmax(action_probability)
            else:
                action = random.randint(0, len(possible_actions)-1)

            new_state, reward, done, info = env.step(possible_actions[action])
            episode_rewards.append(reward)
            new_stacked_frames = add_state(stacked_frames, new_state, False)
            memory.add((format_frames(stacked_frames), action, reward, 
                format_frames(new_stacked_frames), done))

            #env.render()
            if done:
                state = env.reset()
                stacked_frames = add_state(None, state, True)
            else:
                stacked_frames = new_stacked_frames

            sample_batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in sample_batch], ndmin=3)
            actions_mb = np.array([each[1] for each in sample_batch])
            rewards_mb = np.array([each[2] for each in sample_batch]) 
            next_states_mb = np.array([each[3] for each in sample_batch], ndmin=3)
            dones_mb = np.array([each[4] for each in sample_batch])

            next_q_targets = target_model.predict(next_states_mb) 
            targets_mb = np.zeros((batch_size, action_space_size))

            for i in range(batch_size):
                if dones_mb[i] == True:
                    targets_mb[i][actions_mb[i]] = rewards_mb[i]
                else:
                    targets_mb[i][actions_mb[i]] = rewards_mb[i] + discount_rate * np.max(next_q_targets[i])

            policy_model.fit(states_mb, targets_mb, verbose = 0, callbacks = [history])

            tau += 1
            if(tau > max_tau):
                target_model.set_weights(policy_model.get_weights()) 
                tau = 0

        exploration_rate = min_exploration + (max_exploration - min_exploration) * \
            np.exp(-exploration_decay_rate*episode) 

        used_mem = psutil.virtual_memory().used
        print(f"used memory: {used_mem / 1024 / 1024} Mb")

        if((episode + 1) % 5 == 0):
            print('Episode: ' + str(episode + 1) + ' Total Reward: ' + str(np.sum(episode_rewards)))
            policy_model.save('pacman_policy_model.h5')

    plt.plot(history.history['accuracy'])
    env.close()

def play():
    model = models.load_model('pacman_policy_model.h5')
    env = retro.make(game = "PacManNamco-Nes")
    env.frameskip = 2
    possible_actions = [[0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0]]

    state = env.reset()
    stacked_frames = add_state(None, state, True)

    for steps in range(2000):
        action_probability = model.predict(format_frames(stacked_frames).reshape(1, 125, 80, 4))
        action = np.argmax(action_probability)
        new_state, reward, done, info = env.step(possible_actions[action])
        new_stacked_frames = add_state(stacked_frames, new_state, False)

        env.render()

        if done:
            state = env.reset()
            stacked_frames = add_state(None, state, True)
        else:
            stacked_frames = new_stacked_frames

def preprocess_frame(frame):
    gray_frame = rgb2gray(frame)
    cropped_frame = gray_frame[8:-4, 0:-75]
    normalized_frame = cropped_frame/255.0
    resized_frame = transform.resize(normalized_frame, [125, 80])
    #plt.imshow(resized_frame)
    #plt.show()
    return resized_frame

def add_state(stacked_frames, new_state, start_new_episode):
    new_frame = preprocess_frame(new_state)

    if(start_new_episode):
        stacked_frames = deque([np.zeros((125, 80), dtype=np.int) for i in range(4)], maxlen=4)
        for n in range(4):
            stacked_frames.append(new_frame)
    else:
        stacked_frames.append(new_frame)
         
    return stacked_frames

# Class for Policy and Target network
class DeepQNetwork():
    def __init__(self, state_size, output_size, learning_rate):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=state_size))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(output_size))

        self.model.compile(optimizer=optimizers.Adam(learning_rate = learning_rate),
            loss=tf.keras.losses.mean_squared_error,
            metrics=['accuracy'])

# Memory class for storing and sampling experiences
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pushes = 0

    def add(self, experience_tuple):
        if len(self.memory) < self.capacity:
            self.memory.append(experience_tuple)
        else:
            self.memory[self.pushes % self.capacity] = experience_tuple
        self.pushes += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


if __name__ == "__main__":
    #train()
    play()