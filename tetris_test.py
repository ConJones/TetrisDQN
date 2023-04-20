from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import gym_tetris
import imageio
import pygame
import numpy as np
from DQN import DQN

def preprocess_img(img, board):

    # convert to grayscale
    img = img.mean(axis=2)

    # Crop and form bins
    img = img[47:207, 95:175].reshape(20, 160//20, 10, 80//10).mean(-1).mean(1)

    # Set to binary (Block present or not)
    img = img != 0
    board = board != 239

    img = img.astype('int8')

    # Set frozen blocks to 2
    img[board] = 2

    return img

# Create enviorment
env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state_size = (20, 10, 1)
action_size = env.action_space.n
create_render = False
create_video = False
load_agent = "agent_6000+1000+1500"

# DQN Parameters
num_episodes = 10000
num_timesteps = 20000
batch_size = 512
num_screens = 4  # may not be used
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
    if e % 500 == 0 and e != 0:
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
            env.reset()
            state = np.zeros(state_size, dtype=bool)
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
        img, reward, done, info = env.step(action)

        # Progress
        if not done:
            for i in range(20):
                img, reward_temp, done, info = env.step(0)
                #if the episode is done then print the return
                reward += reward_temp
                if done:
                    break

        next_state = preprocess_img(img, info["board"])
        next_state = np.expand_dims(next_state.reshape(20, 10, 1),axis=0)

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

            if create_render or e % 10 == 0:
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