import random
import numpy as np
from collections import deque
import tensorflow as tf
from os.path import exists

class DQN:
    def __init__(self, state_size, action_size, load_agent_name=None):

        #define the state size
        self.state_size = state_size
        
        #define the action size
        self.action_size = action_size
        
        #define the replay buffer
        self.replay_buffer = deque(maxlen=1000)
        
        #define the discount factor
        self.gamma = 0.99  
        
        #define the epsilon value
        self.epsilon = 0.3   
        
        #define the update rate at which we want to update the target network
        self.update_rate = 64*5

        self.load_agent = False if load_agent_name == None else True
        self.load_agent_name = load_agent_name
        
        #define the main network
        self.main_network = self.build_network()
        
        #define the target network
        self.target_network = self.build_network()
        
        #copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())
        

    def build_network(self):
        if self.load_agent and exists(self.load_agent_name):
            model = tf.keras.models.load_model(self.load_agent_name)
        else:
            # Notify if proceeding with new agent and load agent true
            if self.load_agent and not exists(self.load_agent_name):
                print("File not found: Proceeding with new agent")
                
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', input_shape=self.state_size))
            model.add(tf.keras.layers.Activation('relu'))

            model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same'))
            model.add(tf.keras.layers.Activation('relu'))

            # model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
            # model.add(tf.keras.layers.Activation('relu'))
            
            # model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same'))
            # model.add(tf.keras.layers.Activation('relu'))
            
            # model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
            # model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Flatten())


            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
            

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.summary()

        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        Q_values = self.main_network(state)
        
        return np.argmax(Q_values.numpy()[0])

    
    #train the network
    def train(self, batch_size):

        X = []
        Y = []
        
        #sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        self.replay_buffer.clear()
        
        #compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network(next_state)))
            else:
                target_Q = reward
                
            #compute the Q value using the main network 
            Q_values = self.main_network(state).numpy()
            
            Q_values[0][action] = target_Q

            X.append(state[0])
            Y.append(Q_values[0])
            
        #train the main network
        self.main_network.fit(np.array(X), np.array(Y), batch_size=batch_size, epochs=3, verbose=2)

    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())