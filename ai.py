# AI for Doom

# Importing the libraries
import numpy as np # Working with arrays
import torch # An AI is implemented with pytorch
import torch.nn as nn # The model that contains the convolutional layers
import torch.nn.functional as F # The package that contains all the functions we're going to use
import torch.optim as optim # The optimizer
from torch.autograd import Variable # The class that contains dynamic graph which can do very fast comoutations of the gradients

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Making the brain - the convolutional neural network
class CNN(nn.Module):
    # The init function defines the architecture of our convolutional neural network
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        ''' The more we apply some convolutions to the different layers of images, the more we are able to detect some features '''
        ''' For each convolution, in_channels corresponds to the number of channels of its input.  in_channels=1 means the AI is detecting black & white images (1st convo) '''
        ''' out_channels corresponds to the number of features you want to detect for the output. Starting with 32 is a common practice (1st convo) '''
        ''' kernel_size is the dimension of the square that will go through the original image. 5x5 dimension is a common practice for the very 1st convo, so we set it to 5'''
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5) # 1st convolution connection : input -> 1st layer
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # 2nd convolution connection : 1st layer -> 2nd layer
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2) # 3rd convolution connection : 2nd layer -> output
        ''' Flatten all the pixels obtained by different convolutions, to get the huge vector for the input of our neural network '''
        ''' For each full connection, the in_features is the number of input features, which is equal to the number of pixels of the huge vector obtained after flatting all the processd images '''
        ''' The out_features is the number of neurons in the hidden layer. 40 is a good one to start with, we can increase it afterwards. '''
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40) # 1st full connection
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions) # 2nd full connection
    
    # The count neurons function
    def count_neurons(self, image_dim):
        # Create a fake image
        x = Variable(torch.rand(1, *image_dim)) # The * allows us to pass the elements of the image_dim as a list of arguments (since image_dim is currently a tuple)
        # Propagate the image into the neural network
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        # Reach the flattening layer
        ''' Take all the pixels of all the channels and put them 1 after the other into the huge vector which will be the input of the fully connected network '''
        return x.data.view(1, -1).size(1)
    
    # The forward function
    def forward(self, x):
        # Propagate the image into the neural network
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) # 3 is the common choice for kernel size, 2 is the common choice for stride
        # Propagate the signal from the convolutional layers to the hidden layers and then to the output layer
        x = x.view(x.size(0), -1) # Flatten a convolutional layer composed of several channels
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body that can play action using the Softmax activation function
class SoftmaxBody(nn.Module):
    # The init function defines the architecture of our AI
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T # The temperature variable
    
    # The forward function propagates the output signals of the brain to the body of the AI
    def forward(self, outputs):
        ''' There are 7 possibles actions '''
        # Calculate the distribution of probabilities of all possible actions
        probs = F.softmax(outputs * self.T)
        # Sample the final action to play from the above distribution of probabilities
        actions = probs.multinomial()
        return actions

# Making the AI
class AI:
    # The init function
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    # The forward function propagates the signal from the very beginning when the brain getting images to the very end when the body playing action
    def __call__(self, inputs):
        # Convert the image into the right format
        input = Variable(torch.fron_numpy(np.array(inputs, dtype = np.float32)))
        # Get the output signal of the brain
        output = self.brain(input)
        # Put the signal into the body
        actions = self.body(output) # currently in Torch formal
        # Return the actions to play
        return actions.data.numpy()

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions) # brain
softmax_body = SoftmaxBody(T=1.0) # body
ai = AI(brain=cnn, body=softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000) # The memory contains 10000 transitions, and to train the AI, we're gonna sample some mini batches from them

# Implementing Eligibility Trace (n-step Q-Learning)
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    # Iterate through each series in the batch
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        # Compute the cumulative reward from right to left
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        # Get the input and target
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        target.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
    # The init function
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    
    # The add function calculates the cumulative rewards
    def add(self, rewards):
        if isinstance(rewards, list): # If the rewards is the list already
            self.list_of_rewards += rewards
        else: # If the rewards is a single element
            self.list_of_rewards.append(rewards)
        # Maintain the size of elements in list_of_rewards
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    # Compute the average of our list_of_rewards
    def average(self):
        return np.mean(self.list_of_rewards)

ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
     # Each epoch has 200 runs of 10 steps
     memory.run_steps(200)
     # Every batch in a sample batch of 128 steps 
     for batch in memory.sample_batch(128):
         inputs, targets = eligibility_trace(batch)
         # Convert the inputs and targets of the neural network into Torch Variable
         inputs, targets = Variable(inputs), Variable(targets)
         predictions = cnn(inputs)
         # Calculate the loss error
         loss_error = loss(predictions, targets)
         # Back propagate the loss_error to the neural network
         optimizer.zero_grad()
         loss_error.backward()
         # Update the weight
         optimizer.step()
     # Get the rewards steps
     rewards_steps = n_steps.rewards_steps()
     # Add the rewards_steps to the moving average
     ma.add(rewards_steps)
     # Compute the avg reward
     avg_reward = ma.average()
     print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))