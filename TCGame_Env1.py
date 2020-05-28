import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()

    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        curr_state = np.reshape(curr_state, (3,3))
        #check rows
        row_sum = np.sum(curr_state, axis=1)
        
        #check columns
        col_sum = np.sum(curr_state, axis=0)
        
        #check diagonals
        diagonal_sum = [0,0]
                
        for i in range(0, 3):  
            diagonal_sum[0] += curr_state[i][i] 
            diagonal_sum[1] += curr_state[i][3 - i - 1] 
        
        #check if any of them sum to 15
        if (15 in row_sum) or (15 in col_sum) or (15 in diagonal_sum):
            return True
        else:
            return False 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        position = curr_action[0]
        value = curr_action[1]
        curr_state[position] = value
        return (curr_state)


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        reward = 0
        #Status after agent's move
        new_state = self.state_transition(curr_state,curr_action) #get the updated state after agent's move
        terminal = self.is_terminal(new_state) #check if the game is over
        if terminal[0] == True:
            if terminal[1] == 'Win': #agent wins the game
                reward = 10
            else: #it's a tie
                reward = 0
        #Since the game isn't over yet, time to have the env make the move
        else:
            reward = -1 
            actions_avail = list(self.action_space(new_state)[1]) #get the available actions
            env_action = random.choice(actions_avail) #choice a random action
            new_state = self.state_transition(new_state,env_action) #get the updated status of the game after env_action
            terminal = self.is_terminal(new_state)
            if terminal[0] == True:
                if terminal[1] == 'Win': #the env wins
                    reward = -10
                else:
                    reward = 0
            
        return (new_state, reward, terminal)


    def reset(self):
        return self.state
