# gym-maze

The original OSS Readme file has been modified to include the relevant details as to provide context of the game and modifications.

A simple 2D maze environment where an agent (blue dot) finds its way from the top left corner (blue square) to the goal at the bottom right corner (red square). 
The objective is to find the shortest path from the start to the goal.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

### Action space
The agent may only choose to go up, down, right, or left ("N",  "E", "S", "W"). If the way is blocked, it will remain at the same the location. 

### Observation space
The observation space is the (x, y) coordinate of the agent. The top left cell is (0, 0).

### Reward
A reward of 1 is given when the agent reaches the goal. For every step in the maze, the agent recieves a reward of -0.1/(number of cells).

### Note by Fareeha Malik
This reward was changed by me to 0.0 during testing phases, but then I realised it was making the agent not move, as a result I did research online to find out that a negative reward discourages the agent during training from revisiting a cell.

### End condition
The maze is reset when the agent reaches the goal. 

## Maze Versions

### Pre-generated mazes
* 3 cells x 3 cells: _MazeEnvSample3x3_
* 5 cells x 5 cells: _MazeEnvSample5x5_ MAZE USEDby Fareeha Malik
* 10 cells x 10 cells: _MazeEnvSample10x10_
* 100 cells x 100 cells: _MazeEnvSample100x100_

### Randomly generated mazes (same maze every epoch)
* 3 cells x 3 cells: _MazeEnvRandom3x3_
* 5 cells x 5 cells: _MazeEnvRandom5x5_
* 10 cells x 10 cells: _MazeEnvRandom10x10_
* 100 cells x 100 cells: _MazeEnvRandom100x100_

### Randomly generated mazes with portals and loops
With loops, it means that there will be more than one possible path.
The agent can also teleport from a portal to another portal of the same colour. 
* 10 cells x 10 cells: _MazeEnvRandom10x10Plus_
* 20 cells x 20 cells: _MazeEnvRandom20x20Plus_
* 30 cells x 30 cells: _MazeEnvRandom30x30Plus_

## Installation
It should work on both Python 2.7+ and 3.4+. It requires pygame and numpy. 

```bash
cd gym-maze
python setup.py install
```

## Requirements 
- **Pip:** latest version  
- **OS:** Ubuntu (WSL2) or Windows   
- **Dependencies:**
  - `pygame`
  - `stable-baselines3`
  - `gymnasium and NOT gym`
  - `numpy`

## Navigate to 
CSCI4440U_Assignment1\gym-maze-master\gym_maze\src>

**Train, Test/Run and Evaluate**
You can start training direcrly from the command line:

#### Training Commands: (you may modify the timesteps as required)
python train_ppo.py --reward_mode explorer --timesteps 500000

python train_ppo.py --reward_mode survivor  --timesteps 500000

python train_a2c.py --reward_mode explorer -- timesteps 500000

python train_a2c.py --reward_mode survivor --timesteps 500000


#### Commands for visualizing 
python visualize_ppo.py --model_path ./models/ppo_survivor --reward_mode survivor

python visualize_ppo.py --model_path ./models/ppo_explorer --reward_mode explorer

python visualize_a2c.py --model_path ./models/a2c_survivor --reward_mode survivor

python visualize_a2c.py --model_path ./models/a2c_explorer --reward_mode explorer

#### Evaluation commands
python eval_A2C.py --model_path ./models/a2c_explorer --reward_mode explorer

python eval_A2C.py --model_path ./models/a2c_survivor --reward_mode survivor

python eval_PPO.py --model_path ./models/ppo_explorer --reward_mode explorer

python eval_PPO.py --model_path ./models/ppo_survivor --reward_mode survivor


***Arguments References, inspired by the flappy bird class example***

1. "--maze_path", the default maze used from ../envs/maze_samples/maze2d_5x5.npy
2. "--timesteps" the time required to train, default 200 000
3. "--seed", random number pulled from flappy bird class example, defualt 7
4. "--logdir" default="./logs", location at which default logs stored
5. "--modeldir", type=str, default="./models"  location at which default models stored after training
6. "--reward_mode", choice between survivor and explorer, explorer being default. Impleentation for this was not strong, followed examples to help curate my code from flappy bird class example. Idea was to have the agent in maze either survive with focus on fast reach to end and exit of maze, vs exploring agent who had same goal except may explore more and different way to reach end.

**References**
Link to OSS: https://github.com/MattChanTK/gym-maze?tab=readme-ov-file

Several documentations for Stablebaselines3, referenced as used in the code files where used.
