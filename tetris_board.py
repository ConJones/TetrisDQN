from peices import tetriminos
import numpy as np

def get_states(board, piece):

    actions = []
    states = []

    # Set to binary (Block present or not)
    board = board != 239
    board = board.astype('int8')

    height = board_height(board)

    for rot_degree, rot in tetriminos[piece].items():
        for x in range(10-rot.shape[1]+1):
            test = [0]
            y = 20 - height
            while (2 not in test) and y < 21:
                if y < 20: # No need to test if past the bottom row
                    if y-rot.shape[0] +1 >= 0:
                        test = board[y-rot.shape[0]+1:y+1, x:x+rot.shape[1]] + rot
                    else:
                        test = board[0:y+1, x:x+rot.shape[1]] + rot[rot.shape[0]-y-1:,:]
                y += 1
            
            # collision found, go back up 1
            # Go back up an extra 1 for the additional y + 1 at the end of the loop
            y -= 2 
            board_temp = board.copy()
            if y-rot.shape[0] +1 >= 0:
                board_temp[y-rot.shape[0]+1:y+1, x:x+rot.shape[1]] = rot + board_temp[y-rot.shape[0]+1:y+1, x:x+rot.shape[1]]
            else:
                board_temp[0:y+1, x:x+rot.shape[1]] = rot[rot.shape[0]-y-1:,:] + board_temp[0:y+1, x:x+rot.shape[1]]

            actions.append((rot_degree, x))
            states.append(get_props(board_temp))
    
    return actions, np.array(states)


def bumpiness(board):
    """Return the height of the board."""
    total_bumpiness = 0
    min_ys = []

    # Iterate over columns 
    for col in board.T:

        # Find top filled block of the column
        i = 0
        while i < col.shape[0] and col[i] == 0:
            i += 1
        min_ys.append(i)

    for i in range(len(min_ys) - 1):
        total_bumpiness += abs(min_ys[i] - min_ys[i+1])

    return total_bumpiness
    
def num_holes(board):
    """Return the holes in the board."""
    num_holes = 0

    # Iterate over columns 
    for col in board.T:

        # Find top filled block of the column
        i = 0
        while i < len(col) and col[i] == 0:
            i += 1

        # From below top block look for empty blocks below it
        i += 1
        if i < len(col):
            for block in col[i:]:
                if block == 0:
                    num_holes += 1

    # take to sum to determine the height of the board
    return num_holes
    
def board_height(board):
    """Return the height of the board."""
    # look for any piece in any row
    # take to sum to determine the height of the board
    return board.any(axis=1).sum()

def cleared_lines(board):
    """Return the height of the board."""
    lines = 0
    for e in board.sum(axis=1):
        if e == 10:
            lines += 1
    return lines

def get_props(board):
    return np.array([cleared_lines(board), board_height(board), num_holes(board), bumpiness(board)])
    #return np.array([cleared_lines(board), board_height(board), num_holes(board)])