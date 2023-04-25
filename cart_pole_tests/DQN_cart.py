import random
import numpy as np
from collections import deque
import tensorflow as tf

class DQN:
    def __init__(self, state_size, action_size, load_agent_name=None):

        #define the state size
        self.state_size = state_size
        
        #define the action size
        self.action_size = action_size
        
        #define the replay buffer
        self.replay_buffer = deque(maxlen=200)
        
        #define the discount factor
        self.gamma = 0.9  
        
        #define the epsilon value
        self.epsilon = 0.0   
        
        #define the update rate at which we want to update the target network
        self.update_rate = 100

        self.load_agent = False if load_agent_name == None else True
        self.load_agent_name = load_agent_name
        
        #define the main network
        self.main_network = self.build_network()
        
        #define the target network
        self.target_network = self.build_network()
        
        #copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())
        

    def build_network(self):
        if self.load_agent:
            model = tf.keras.models.load_model(self.load_agent_name)
        else:
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(100, input_dim=4, 
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')))
            model.add(tf.keras.layers.Dense(50,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')))
            model.add(tf.keras.layers.Dense(
                self.action_size,
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2)))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.summary()

        return model


    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        Q_values = self.main_network(state).numpy()
        
        return np.argmax(Q_values[0])

    
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
        self.main_network.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=2)

    #update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())