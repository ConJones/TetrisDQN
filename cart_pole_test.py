from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import imageio
import pygame
import numpy as np
from DQN_cart import DQN
import matplotlib.pyplot as plt

import gym

# Create enviorment
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_size = (4)
action_size = env.action_space.n
create_render = True
create_video = False
load_agent = "agent_cart+3000"

# DQN Parameters
num_episodes = 10000
batch_size = 4
dqn = DQN(state_size, action_size, load_agent)

# Create outputs
#surface = pygame.display.set_mode((600, 400), pygame.DOUBLEBUF)
filename = "test_tet"
filename = filename + ".mp4"
#video = imageio.get_writer(filename, fps=30)

target_update_step = 0
done = True
for e in range(num_episodes):
    # Reset Steps
    total_reward = 0
    time_step = 0
    
    # Save checkpoints
    if e % 1000 == 0:
        dqn.main_network.save(load_agent + '+' + str(e))

    # Evaluate every 10 episodes
    if e % 10 == 0:
        temp_eps = dqn.epsilon
        dqn.epsilon = 0
    else:
        dqn.epsilon = temp_eps

    while True:
        # Reset if finished
        if done:
            state = env.reset()
            state = np.expand_dims(state,axis=0)

        # Increment timestep 
        target_update_step += 1
        time_step += 1

        # Update the target network
        if target_update_step % dqn.update_rate == 0:
            target_update_step = 0
            dqn.update_target_network()

        # Select action
        action = dqn.epsilon_greedy(state)

        # Preform action and process board
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state,axis=0)

        # Store transition
        dqn.store_transistion(state, action, reward, next_state, done)

        # Update state for next step
        state = next_state

        #update the return
        total_reward += reward

        if e % 100 == 0 or create_render:
            env.render()
        
        # if create_render or create_video or e % 10 == 0:
        #     img = env.render(mode="rgb_array")
        #     img = np.transpose(img, (1,0,2))
            
        #     if create_video:
        #         video.append_data(img)

        #     if create_render or e % 10 == 0:
        #         pygame.pixelcopy.array_to_surface(surface, img)
        #         pygame.display.flip()
        
        #if the episode is done then print the return
        if done:
            print('Episode: ',e, ',' 'Return', total_reward, ', Lasted: ', time_step, ' steps')
            break

        #if the number of transistions in the replay buffer is greater than batch size
        #then train the network
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)

env.close()
#video.close()