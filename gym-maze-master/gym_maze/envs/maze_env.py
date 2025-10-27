import numpy as np

#old gym REMOVED assertionerror
#import gym
#from gym import error, spaces, utils
#from gym.utils import seeding


import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from gym_maze.envs.maze_view_2d import MazeView2D #when decoupling recheck this


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    
    ACTION = ["N", "E", "S", "W"] #correct order for gymnasium
    

#adding personas via rewards clogic (personas_reward)
    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True, reward_design = "" ):

        self.viewer = None
        self.enable_render = enable_render
        self.reward_design = reward_design #adding for reward design req (recheck i think mistake)

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gymnasium - Maze (%s)" % maze_file, #modified the window name
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gymnasium - Maze (%d x %d)" % maze_size, #modifed window name
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size


        self.action_space = spaces.Discrete(len(self.ACTION)) #for 4 options N E S W
        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


#TODO modify for new gymanasium versus gym
    def step(self, action):

        self.current_step += 1
    
    #robot/blue ball/agent movement
        dir = self.ACTION[action]
        self.maze_view.move_robot(dir)



        #https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        
        #ISSUE FOR EVALUATION, because checking for exact rocbot coordinates to reach goal
        #check for goal
        terminated = np.array_equal(self.maze_view.robot, self.maze_view.goal)
        #truncated = False
        truncated = self.current_step >= self.max_steps

        current_cell = tuple(self.maze_view.robot)

#reward logic, applicable to eval
        if self.reward_design == "survivor":
            reward = 1.0 if terminated else -0.1/(self.maze_size[0]*self.maze_size[1])
        
        elif self.reward_design == "explorer":
             #adding to helpeval
             if terminated:
          #  self.done = True #forcing done when the ball reaches end of maze (even if looping again and again up and down
                reward = 1.0 #fixed, csv issue (1.0 meaning it gets reward for goal)
             elif current_cell not in self.visited_cells:
              reward = 0.5 #rewarding for first visit of cell
             else: 
              reward =-0.1 #explorer keep going up and down
        else: 
                reward = 1.0 if terminated else 0.0 #no reward if more than 1st visit of cell
#reward logic (assignment design method using survivor and explorer)
        #current_cell = tuple(self.maze_view.robot)
            #current_cell = tuple(self.maze_view.robot) MISPLACED
            #if current_cell not in self.visited_cells: #self.maze_view.visited_cells
            
       # self.maze_view.visited_cells.add(current_cell) 
        self.visited_cells.add(current_cell)  #tracking visited cells by survivor or explorer persona

        self.state = self.maze_view.robot.copy() #updating state every step
    
    #adding info to be returned
    #retrace github, did i remove this, remeber having
        info = {
            "visited_cells": len(self.visited_cells),
            "total_cells": self.maze_size[0] * self.maze_size[1], 
            "goal_reached": terminated #debugging, csv always 0 in success
        }

       # return self.state, reward, done, info #TODO stable-Baselines3 issue (check return documentaiton)
        return self.state, reward,  terminated, truncated, info #ordering check

#typeerror, adding seed and options (to match gymnasium)
#TODO modify!!
    def reset(self, *, seed = None, options = None):
        self.seed(seed)
        self.maze_view.reset_robot()
        #self.state = np.zeros(2)
        self.visited_cells = set() #the cells the robot/blue ball visit
        self.visited_cells.add(tuple(self.maze_view.robot)) #cell 0 for the agent
        self.state = self.maze_view.robot.copy()
        self.steps_beyond_done = None
        self.done = False
        self.current_step = 0
        self.max_steps = 1000 #placeholder size, till csv works
        info={
            "visited_cells": len(self.visited_cells),
            "total_cells": self.maze_size[0] * self.maze_size[1], 
            "goal_reached": False}
        return self.state,info #added info (new documentation stablebaselines)

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)
