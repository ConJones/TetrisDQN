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
        self.replay_buffer = deque(maxlen=1024)
        
        #define the discount factor
        self.gamma = 0.8  
        
        #define the epsilon value
        self.epsilon = 0   
        
        #define the update rate at which we want to update the target network
        self.update_rate = 5000

        self.learning_rate = .001

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
            model.add(tf.keras.layers.Dense(32, input_dim=self.state_size, 
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')))
            model.add(tf.keras.layers.Dense(64,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')))
            model.add(tf.keras.layers.Dense(
                self.action_size,
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2)))
            

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        model.summary()

        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        Q_values = self.main_network(state).numpy()
        
        return np.argmax(Q_values)

    
    #train the network
    def train(self, batch_size):

        X = []
        Y = []
        
        #sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        #compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Qs = self.target_network(next_state).numpy()
                target_Q = (reward + self.gamma * np.amax(target_Qs))
            else:
                target_Q = reward
                
            #compute the Q value using the main network 
            Q_values = self.main_network(state).numpy()
            
            Q_values[0][action] = target_Q

            X.append(state[0])
            Y.append(Q_values[0])
            
        #train the main network
        self.main_network.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=2)

    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())