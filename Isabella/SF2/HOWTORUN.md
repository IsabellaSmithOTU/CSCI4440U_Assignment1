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

7. **Evaluation**
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

8. **TroubleShooting**
| Issue                       | Cause                           | Fix                                                          |
| --------------------------- | ------------------------------- | ------------------------------------------------------------ |
| No visuals during training  | No render mode                  | Add `render=True` when creating env on line 256 of train_SF2.py |
| TensorBoard not updating    | Logs not found                  | Confirm `--log-dir` matches TensorBoard path                 |


**References**
https://medium.com/aureliantactics/integrating-new-games-into-retro-gym-12b237d3ed75

https://stable-retro.farama.org/index.html

https://github.com/openai/retro