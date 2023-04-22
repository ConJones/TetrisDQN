from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import gym_tetris
import imageio
import pygame
import numpy as np
from DQN import DQN
from math import floor

def preprocess_img(img, board):

    # convert to grayscale
    img = img.mean(axis=2)

    # Crop and form bins
    img = img[47:207, 95:175].reshape(20, 160//20, 10, 80//10).mean(-1).mean(1)

    # Set to binary (Block present or not)
    img = img != 0
    board = board != 239

    # find board height
    # look for any piece in any row
    board_1D = board.any(axis=1)
    # take to sum to determine the height of the board
    height =  board_1D.sum()

    img = img.astype('int8')

    # Set frozen blocks to 2
    img[board] = 2

    img = np.concatenate((img[0:2,:], np.zeros((1, 10), dtype=np.int8), img[20-height:(20-height+4 if 20-height+4 <= 20 else 20),:]), axis=0)

    if img.shape[0] < 7:
        img = np.concatenate((img, 2*np.ones((7-img.shape[0], 10), dtype=np.int8)))

    return img

# Create enviorment
env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state_size = (7, 10, 1)
action_size = 40
create_render = False
create_video = False
evaluate = 20
load_agent = "agent_7x10_nr_new_new_net"

# DQN Parameters
num_episodes = 100000
num_timesteps = 20000
batch_size = 64
dqn = DQN(state_size, action_size, load_agent)

# Create outputs
surface = pygame.display.set_mode((256, 240), pygame.DOUBLEBUF)
filename = "test_tet"
filename = filename + ".mp4"
video = imageio.get_writer(filename, fps=30)

target_update_step = 0
done = True
for e in range(num_episodes):
    # Reset Steps
    total_reward = 0
    time_step = 0
    
    # Save checkpoints
    if e % 1000 == 0 and e != 0:
        dqn.main_network.save(load_agent + '+' + str(e))

    # Evaluate every 10 episodes
    if e % evaluate == 0:
        temp_eps = dqn.epsilon
        dqn.epsilon = 0
    else:
        dqn.epsilon = temp_eps

    while True:
        # Reset if finished
        if done:
            env.reset()
            img, reward, done, info = env.step(0)
            state = preprocess_img(img, info["board"])
            state = np.expand_dims(state.reshape(7, 10, 1),axis=0)

        # Increment timestep 
        target_update_step += 1
        time_step += 1

        # Update the target network
        if target_update_step % dqn.update_rate == 0:
            target_update_step = 0
            dqn.update_target_network()

        # Select action
        action = dqn.epsilon_greedy(state)

        # convert from action number to rotate and x movement
        rotation = action % 4
        x_pos = floor(action / 4)

        try:
            reward = 0
            # Preform action and process board
            if(rotation == 1):
                # Rotate 90
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)
            elif(rotation == 2):
                # Rotate 180
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)
            elif(rotation == 3):
                # Rotate -90
                img, _, done, info = env.step(2)
                img, _, done, info = env.step(0)

            if x_pos - 5 < 0:
                for i in range(5 - x_pos):
                    # Move Left
                    img, _, done, info = env.step(4)
                    img, _, done, info = env.step(0)
            else:
                for i in range(x_pos - 5):
                    # Move Right
                    img, _, done, info = env.step(3)
                    img, _, done, info = env.step(0)

            pre_drop_stats = info["statistics"].copy()
            while(info["statistics"] == pre_drop_stats):
                # Drop until frozen
                img, reward_temp, done, info = env.step(5)
                if reward_temp != 0:
                    reward = reward_temp
            img, _, done, info = env.step(0)

        except ValueError:  
            done = True

        next_state = preprocess_img(img, info["board"])
        next_state = np.expand_dims(next_state.reshape(7, 10, 1),axis=0)

        # Store transition
        dqn.store_transistion(state, action, reward, next_state, done)

        # Update state for next step
        state = next_state

        #update the return
        total_reward += reward
        
        if create_render or create_video or e % 10 == 0:
            img = env.render(mode="rgb_array")
            img = np.transpose(img, (1,0,2))
            
            if create_video:
                video.append_data(img)

            if create_render or e % evaluate == 0:
                pygame.pixelcopy.array_to_surface(surface, img)
                pygame.display.flip()
        
        #if the episode is done then print the return
        if done:
            print('Episode: ',e, ',' 'Return', total_reward, ', Lasted: ', time_step, ' steps')
            break

        #if the number of transistions in the replay buffer is greater than batch size
        #then train the network
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)

env.close()
video.close()