#  How to Run the Street Fighter II Deep Reinforcement Learning Project

This guide explains how to install, configure, and run the DRL training pipeline for **Street Fighter II: Special Champion Edition** using a modified version of **OpenAI Stable-Retro**.

---

##  Requirements

- **Python:** 3.8–3.10  
- **Pip:** latest version  
- **OS:** Ubuntu (WSL2) or Windows   
- **Dependencies:**
  - `stable-retro` (modified)
  - `pygame`
  - `stable-baselines3`
  - `torch`
  - `numpy`

---

##  Setup (Ubuntu or WSL)

1. **Clone the repository**
   ```bash
   git clone https://github.com/IsabellaSmithOTU/CSCI4440U_Assignment1
   cd Isabella/SF2
   ```
If you are running on WSL then run this command:
```
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip ffmpeg -y
```
    If you are running on Windows:
    1. Install python
    2. Install FFmpeg and OpenGL

2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv retro_env
    source retro_env/bin/activate (WSL)
    retro_env\Scripts\activate (Windows)
    ```
3. **Install dependencies**
```
    pip install -r requirements.txt
```
4. **Import ROM**
```
    python -m retro.import /(exact path)/CSCI4440U_Assignment1/Isabella/SF2
```

Verify that its imported:

 ```
 python -m retro.list
```

Output should be:
```
StreetFighterIISpecialChampionEdition-Genesis
```

5. **Modify Data file**

The Street Fighter Special Champion edition has been slightly modified for training purposes.

Navigate to: 
```
/retro_env/lib/python3.10/site-packages/retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis
```
And replace the data.json in this folder with the data.json in the repository's  `/retro_assets`

```
StreetFighterIISpecialChampionEdition-Genesis/
│
├── rom.md
├── rom.sha
├── data.json -> REPLACE THIS FILE
├── metadata.json
├── scenario.json
└── Champion.Level1.RyuVsGuile.state
```

6. **Train and Evaluate**
You can start training direcrly from the command line:
Example: PPO, Brute Force Persona with 500k Timesteps
```
** Navigate to /SF2
python train_SF2.py --algo ppo --persona brute_force --game StreetFighterIISpecialChampionEdition-Genesis --state Champion.Level1.RyuVsGuile --timesteps 500000
```
Alternate Example: A2C, Survival Persona
```
** Navigate to /SF2
python train_SF2.py --algo a2c --persona survival --timesteps 1000000
```
***Arguments Reference***
| Argument      | Description                                        | Default                                         |
| ------------- | -------------------------------------------------- | ----------------------------------------------- |
| `--algo`      | Algorithm to use (`ppo` or `a2c`)                  | `ppo`                                           |
| `--persona`   | Reward-shaping persona (`brute_force`, `survival`) | `brute_force`                                   |
| `--game`      | Game ID for Gym Retro                              | `StreetFighterIISpecialChampionEdition-Genesis` |
| `--state`     | State file within the game’s folder                | `Champion.Level1.RyuVsGuile`                    |
| `--timesteps` | Total training timesteps                           | `1_000_000`                                     |
| `--seed`      | Random seed                                        | `42`                                            |
| `--n_envs`    | Number of parallel environments                    | `4`                                             |
| `--model-dir` | Directory for saved models                         | `./sf2_models`                                  |
| `--log-dir`   | Directory for TensorBoard logs                     | `./sf2_logs`                                    |

You can monitor live training through TensorBoard logs. TensorBoard logs are stored in `sf2_logs/`.

To view training curves enter this in your console:
```
tensorboard --logdir=sf2_logs
```
Then open your browser and visit: http://localhost:6006

**TroubleShooting**
| Issue                       | Cause                           | Fix                                                          |
| --------------------------- | ------------------------------- | ------------------------------------------------------------ |
| No visuals during training  | No render mode                  | Add `render=True` when creating env on line 256 of train_SF2.py |
| TensorBoard not updating    | Logs not found                  | Confirm `--log-dir` matches TensorBoard path                 |

**References**
#  How to Run the Street Fighter II Deep Reinforcement Learning Project

This guide explains how to install, configure, and run the DRL training pipeline for **Street Fighter II: Special Champion Edition** using a modified version of **OpenAI Gym Retro**.

---

##  Requirements

- **Python:** 3.8–3.10  
- **Pip:** latest version  
- **OS:** Ubuntu (WSL2) or Windows   
- **Dependencies:**
  - `stable-retro` (modified)
  - `pygame`
  - `stable-baselines3`
  - `torch`
  - `numpy`
  - `pygame` (for rendering)

---

##  Setup (Ubuntu or WSL)

1. **Clone the repository**
   ```bash
   git clone https://github.com/IsabellaSmithOTU/CSCI4440U_Assignment1
   cd Isabella/SF2
   ```
If you are running on WSL then run this command:
```
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip ffmpeg -y
```
    If you are running on Windows:
    1. Install python
    2. Install FFmpeg and OpenGL

2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv retro_env
    source retro_env/bin/activate (WSL)
    retro_env\Scripts\activate (Windows)
    ```
3. **Install dependencies**
```
    pip install -r requirements.txt
```
4. **Import ROM**
```
    python -m retro.import /(exact path)/CSCI4440U_Assignment1/Isabella/SF2
```

Verify that its imported:

 ```
 python -m retro.list
```

Output should be:
```
StreetFighterIISpecialChampionEdition-Genesis
```

5. **Modify Data file**

The Street Fighter Special Champion edition has been slightly modified for training purposes.

Navigate to: 
```
/retro_env/lib/python3.10/site-packages/retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis
```
And replace the data.json in this folder with the data.json in the repository's  `/retro_assets`

```
StreetFighterIISpecialChampionEdition-Genesis/
│
├── rom.md
├── rom.sha
├── data.json -> REPLACE THIS FILE
├── metadata.json
├── scenario.json
└── Champion.Level1.RyuVsGuile.state
```

6. **Training**
You can start training direcrly from the command line:
Example: PPO, Brute Force Persona with 500k Timesteps
```
** Navigate to /SF2
python train_SF2.py --algo ppo --persona brute_force --game StreetFighterIISpecialChampionEdition-Genesis --state Champion.Level1.RyuVsGuile --timesteps 500000
```
Alternate Example: A2C, Survival Persona
```
** Navigate to /SF2
python train_SF2.py --algo a2c --persona survival --timesteps 1000000
```
**Arguments Reference**
| Argument      | Description                                        | Default                                         |
| ------------- | -------------------------------------------------- | ----------------------------------------------- |
| `--algo`      | Algorithm to use (`ppo` or `a2c`)                  | `ppo`                                           |
| `--persona`   | Reward-shaping persona (`brute_force`, `survival`) | `brute_force`                                   |
| `--game`      | Game ID for Gym Retro                              | `StreetFighterIISpecialChampionEdition-Genesis` |
| `--state`     | State file within the game’s folder                | `Champion.Level1.RyuVsGuile`                    |
| `--timesteps` | Total training timesteps                           | `1_000_000`                                     |
| `--seed`      | Random seed                                        | `42`                                            |
| `--n_envs`    | Number of parallel environments                    | `4`                                             |
| `--model-dir` | Directory for saved models                         | `./sf2_models`                                  |
| `--log-dir`   | Directory for TensorBoard logs                     | `./sf2_logs`                                    |

You can monitor live training through TensorBoard logs. TensorBoard logs are stored in `sf2_logs/`.

To view training curves enter this in your console:
```
tensorboard --logdir=sf2_logs
```
Then open your browser and visit: http://localhost:6006

7. **Environments: Actions, Observations, and Rewards** 

The experimental environment was constructed using the Retro framework wrapped in Gymnasium, enabling a reinforcement learning interface for Street Fighter II. Each model interacted with the game environment through discrete actions, observed game states, and received scalar rewards that guided learning.

#### Actions
The action space was discrete, representing combinations of 12 player inputs such as movement and attacks.
Key mapped actions included:
- Movement: left, right, crouch, jump
- Attack types: punch, kick, special move
- Combinations: e.g., crouch + kick, jump + punch

#### Observation
The observation space consisted of preprocessed image frames from the emulator, capturing the in-game state at each timestep. Each frame was:
- Stacked over several frames to provide temporal context (frame history),
- Normalized to improve training stability.
These observations allowed the agent’s convolutional neural network to infer spatial relationships between fighters, projectiles, and health bars.
In addition to this, multiple environments were ran parallely to increase the speed of the training (`n_envs`)

The metrics observed in training were:
1. Player's Health
2. Opponent's Health
3. Player score
4. Player's x position
5. Opponent's x position
These values were through through the stable-retro API's data.

The most notable metrics observed in evaluation were:
1. Average reward given to agent
2. Number of Steps
3. Matches Won
4. Matches Lost
5. Time Lasted

#### Rewards 
Distinct reward structures were implemented to shape agent behavior into specialized combat “personas.” The reward function was central to determining strategic tendencies, balancing aggression and survival instincts across training sessions.

##### Brute Force Persona
This persona emphasized offensive play, rewarding aggressive engagement and high-damage attacks. To avoid degenerate behavior (e.g., inactivity), the function incorporated both positive and negative incentives:

- (+) Damage Dealt Reward: Encouraged the agent to inflict maximum damage on the opponent per frame.

- (−) Damage Taken Penalty: Applied when the agent received damage, but kept moderate in magnitude. A smaller penalty prevented the model from converging to the local optimum of avoiding combat by standing still.

- (+) Score-Based Bonus: Provided an additional reward for executing successful combos or heavy-hitting attacks, promoting varied offensive tactics.

- (−) Inactivity Penalty: Discouraged periods of no movement or attack, ensuring continuous engagement during matches.

Overall, this configuration produced highly aggressive behavior patterns with limited defensive adaptation.

##### Survival Persona

This persona prioritized defensive strategy and endurance, training the agent to maintain distance, avoid damage, and outlast opponents. The reward function was designed accordingly:

- (+) Consistent Survival Reward: Small continuous positive reward for each frame survived, encouraging sustained engagement without being defeated.

- (−) Damage Taken Penalty: Strong penalty proportional to health loss, reinforcing evasion and blocking behavior.

- (+) Distance Maintenance Reward: Reward for maintaining an optimal distance from the opponent, promoting strategic positioning and spatial awareness.

The survival persona’s structure resulted in defensive, evasive agents capable of withstanding longer bouts but often less capable of finishing matches quickly.

8. **Evaluation**
The evaluation script (eval_SF2.py) runs your trained model in the game environment and measures:
- Episode rewards
- Match wins/losses
- Step count
- Win rate
- Persona-based shaped reward

Example:
```
python eval_SF2.py \
  --model_path models/ppo_sf2.zip \
  --game StreetFighterIISpecialChampionEdition-Genesis \
  --state Champion.Level1.RyuVsGuile \
  --persona brute_force \
  --episodes 5 \
  --render 1 \
  --csv_out logs/sf2_eval_metrics.csv
```
**Arguments Reference**
| Argument       | Description                                | Default                                         |
| -------------- | ------------------------------------------ | ----------------------------------------------- |
| `--model_path` | Path to the trained PPO/A2C model (`.zip`) | **required**                                    |
| `--game`       | Name of the Retro-compatible game          | `StreetFighterIISpecialChampionEdition-Genesis` |
| `--state`      | Game state or level to load                | `Champion.Level1.RyuVsGuile`                    |
| `--persona`    | Reward-shaping persona                     | `brute_force`                                   |
| `--episodes`   | Number of evaluation episodes              | `10`                                            |
| `--render`     | `1` = show gameplay window, `0` = headless | `0`                                             |
| `--csv_out`    | Path to save evaluation metrics            | `logs/sf2_eval_metrics.csv`                     |

9. **Reproduction**
```
python train_SF2.py --timesteps 500000
```

```
python train_SF2.py --algo a2c --timesteps 500000
```
```
python train_SF2.py --persona survival --timesteps 500000
```
8. **TroubleShooting**

| Issue                       | Cause                           | Fix                                                          |
| --------------------------- | ------------------------------- | ------------------------------------------------------------ |
| No visuals during training  | No render mode                  | Add `render=True` when creating env on line 256 of train_SF2.py |
| TensorBoard not updating    | Logs not found                  | Confirm `--log-dir` matches TensorBoard path                 |


**References**

https://medium.com/aureliantactics/integrating-new-games-into-retro-gym-12b237d3ed75

https://stable-retro.farama.org/index.html

https://github.com/openai/retro