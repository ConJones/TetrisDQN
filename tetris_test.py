from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import gym_tetris
import imageio
import pygame
import numpy as np
from DQN import DQN
from math import floor
import tetris_board
from peices import starting_width
from math import log10


# Create enviorment
env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state_size = 4
action_size = 1
create_render = True
create_video = False
evaluate = 20
load_agent = "agent_no_conv_big_delta_state"
max_total_reward = 0

# DQN Parameters
num_episodes = 100000
num_timesteps = 20000
batch_size = 512
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
    if e % 1000 == 0:
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
            button_inputs, board_state = tetris_board.get_states(info["board"], info["current_piece"][0])
            state = board_state
            # Normalize Data
            state = state.astype(float)
            state[:,0] = (1.5 ** state[:,0])/(1.5 ** 4)
            state[:,1:3] = np.clip((-state[:,1:3]/4) + 1, -1, 1)
            state[:,3] = -np.log10(np.maximum(state[:,3], -0.5) + 1.5) + 1

        # Increment timestep 
        target_update_step += 1
        time_step += 1

        # Update the target network
        if target_update_step % dqn.update_rate == 0:
            target_update_step = 0
            dqn.update_target_network()

        # Select action
        dqn.action_size = len(state)
        action = dqn.epsilon_greedy(state)
        button_inputs_for_action = button_inputs[action]

        # convert from action number to rotate and x movement
        rotation = button_inputs_for_action[0]
        try:
            reward = 0
            # Preform action and process board
            if(rotation == 90):
                # Rotate 90
                img, _, done, info = env.step(2)
                img, _, done, info = env.step(0)
            elif(rotation == 180):
                # Rotate 180
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)
            elif(rotation == 270):
                # Rotate -90
                img, _, done, info = env.step(1)
                img, _, done, info = env.step(0)

            x_pos = button_inputs_for_action[1] - 5 + starting_width[info["current_piece"]][0]
            if x_pos < 0:
                for i in range(-x_pos):
                    # Move Left
                    img, _, done, info = env.step(4)
                    img, _, done, info = env.step(0)
            else:
                for i in range(x_pos):
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

        button_inputs, next_board_state = tetris_board.get_states(info["board"], info["current_piece"][0])
        # Store transition
        # action 0 because we only need to update the 1 state we are passing in
        next_state = next_board_state.copy()
        next_state[:,1:4] = next_board_state[:,1:4] - board_state[action, 1:4]
        reward += 4

        # Normalize Data
        next_state = next_state.astype(float)
        next_state[:,0] = (1.5 ** next_state[:,0])/(1.5 ** 4)
        next_state[:,1:3] = np.clip((-next_state[:,1:3]/4) + 1, -1, 1)
        next_state[:,3] = -np.log10(np.maximum(next_state[:,3], -0.5) + 1.5) + 1

        dqn.store_transistion(np.expand_dims(state[action],axis=0), 0, reward, next_state, done)

        # Update state for next step
        state = next_state
        board_state = next_board_state

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
    
    if total_reward > max_total_reward:
        max_total_reward = total_reward
        dqn.main_network.save(load_agent + '+' + str(e) + '_TR' + str(max_total_reward) )

    #if the number of transistions in the replay buffer is greater than batch size
    #then train the network
    if len(dqn.replay_buffer) >= batch_size:
        dqn.train(batch_size)

env.close()
video.close()