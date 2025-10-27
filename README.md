# CSCI4440U_Assignment1
DRL for Automated Testing
This project explores the use of Deep Reinforcement Learning (DRL) to train intelligent agents capable of playing two games - Street Fighter II and a 2D Maze Environment. Using the Stable Baselines3 framework with PPO and A2C algorithms, agents were trained in a custom Gymnasium-compatible environment built with Retro and Pygame wrappers.

## Street Fighter II

### Street Fighter II - Special Champions Edition
![Alt Text](https://s5.ezgif.com/tmp/ezgif-573e62cb23cc68.gif)

A custom Gymnasium-compatible environment was implemented using the Retro emulator framework and Pygame interface, enabling full integration with Stable Baselines3. Three trained models were compared:

1. PPO (Brute Force) – optimized for aggressive play and high attack frequency,

2. A2C (Brute Force) – similar offensive strategy using an alternative algorithmic approach,

3. PPO (Survival) – focused on defensive strategies and maximizing episode longevity.

#### Methodology
Agents were trained for 500,000 timesteps each under identical environmental and preprocessing conditions. Custom reward functions were designed to encourage target behaviors, including damage dealt, round victories, and sustained survival time. Model evaluation was conducted through repeated episode simulations, generating quantitative metrics such as cumulative reward, match win rate, and episode duration.

#### Results and Analysis
Post-training evaluations revealed that PPO (Brute Force) achieved the highest average match win rate (~36.7%), followed by A2C (Brute Force) and PPO (Survival) (each ~31.7%). PPO-based models demonstrated more consistent learning dynamics and superior episode stability compared to A2C.
Comprehensive performance visualization was performed using Matplotlib, presenting learning curves, metric distributions, and comparative trend analyses across models and personas.
#### Conclusion
The experimental findings confirm that PPO provides more stable and effective policy optimization for complex fighting environments than A2C under identical training conditions. Reward shaping was found to significantly influence behavioral outcomes, supporting the viability of persona-based learning in reinforcement-driven game AI systems.

### Street Fighter II Demo
More information in `Isabella/SF2`
https://youtu.be/L3RfJb5U9sQ

## Street Fighter II

### 2D Maze game - Retrieved from OSS)
Link to OSS: https://github.com/MattChanTK/gym-maze?tab=readme-ov-file

A custom gym environment was implemented using the Pygame and numpy interface, enabling full integration with Stable Baselines3. Four trained models were compared:

1. PPO (Explorer),
2. PPO (Survivor),
3. A2C (Explorer),
4. A2C (Survivor)

#### Methodology
Agents were trained for 500,000 timesteps each under identical environmental and preprocessing conditions. Custom reward functions were designed to encourage target behaviors, which focussed on getting from point A (blue cell-start of maze) to point B (Red call-end of maze) at which point the game woulc conclude. Due to the several avaliable maze environments, and the limited time with a focus on ensuring model implementation was ready, the 5x5 model maze was used. Model evaluation was conducted through repeated episode simulations, generating quantitative metrics such as episode numbers (with a limit to 10 due to implementation ease), mean reward achieved,steps,success (1 for True, 0 for False), success rate, truncated (1 for True, 0 for False) and coverage/mean across the maze.

#### Results and Analysis
Post-training evaluations revealed that all 4 experiments achieved a 100% success rate (meaning the blue ball/agent successfully reached from Point A to Point B, the exit of the maze. PPO-based models demonstrated more consistent learning dynamics and superior episode stability compared to A2C.
Comprehensive performance visualization was performed using Matplotlib, presenting learning curves, Success/Truncated per episode, and coverage per episode across models and personas. Due to lack of longer episodes as a result of model breakage, and prioritizing the models working, the data for the visualizations was not intensive enough to provide a comprehensive analysis. 

#### Conclusion
The experimental findings confirm that PPO is a faster DRL model for agents to learn and accurately fulfill requirements. Due to lack of further data, deeper analysis could not be made as the A2C model, required several fixes after continuous model breakages, presenting a lack of time for an in-depth qualitative analysis. However, it is important to note that across the many attempts at training, PPO always maintained a faster speed, training at half the time compared to A2C. Improvements in a future project would be made, including deeper qualitative analysis and a prioritization of training 3 models only, rather than going beyond the minimum requirement.

### 2D maze Demo
More information in `

