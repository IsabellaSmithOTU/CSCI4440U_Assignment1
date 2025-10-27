# CSCI4440U_Assignment1
DRL for Automated Testing
This project explores the use of Deep Reinforcement Learning (DRL) to train intelligent agents capable of playing two games - Street Fighter II and a 2D Maze Environment. Using the Stable Baselines3 framework with PPO and A2C algorithms, agents were trained in a custom Gymnasium-compatible environment built with Retro and Pygame wrappers.

##Street Fighter II

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
