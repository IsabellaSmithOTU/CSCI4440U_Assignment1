import retro
import gymnasium as gym
import argparse
import os
import numpy as np
import tensorboard
import traceback
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecNormalize

#STREET FIGHTER II DRL TRAINING
"""
ACTION SPACE NOTES:

Action space is defined within the Retro environment using FILTERED actions. (Not explicitly defined here)
The following code is the Genesis controller mapping for reference taken from Retro's data for its Genesis Emulator:
"Genesis": {
        "lib": "genesis_plus_gx",
        "ext": ["md"],
        "rambase": 16711680,
        "keybinds": ["X", "Z", "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT", "C", "A", "S", "D"],
        "buttons": ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"],
        "types": ["|u1", ">u2", ">u4", "|i1", ">i2", ">i4", "|d1", ">d2", ">d4", "<d4", ">d6", ">d8", ">n4", ">n6", ">n8"],
        "overlay": ["=", ">", 2],
        "actions": [
            [[], ["UP"], ["DOWN"]],
            [[], ["LEFT"], ["RIGHT"]],
            [[], ["A"], ["B"], ["C"], ["X"], ["Y"], ["Z"], ["A", "B"], ["B", "C"], ["A", "X"], ["B", "Y"], ["C", "Z"], ["X", "Y"], ["Y", "Z"]]
        ]
    }
    I tested these buttons in the test_env/test.py file to identify which buttons correspond to which actions in Street Fighter II.

"""

# Wrapper Class 
class StreetFighterEnv(gym.Wrapper):
    """
    Custom environment wrapper for Street Fighter II, implementing the 
    'brute_force' and 'strategist' reward personas.
    The base environment (retro.make) is created directly within __init__.
    """
    def __init__(self, game: str, state: str, persona: str, player: int = 1, render_mode: str = None):
        
        # Create the base Retro environment
        env = retro.make(
            game=game, 
            state=state, 
            render_mode=render_mode,
            use_restricted_actions=retro.Actions.FILTERED
        )
        
        super().__init__(env)
        
        # Store the selected persona
        self.persona = persona
        self.player = player
        self.opponent_id = 3 - player
        
        # Initialize tracking variables
        self.previous_action = None
        self.previous_distance = 0.0
        
        # Track cumulative custom reward for logging
        self.episode_custom_reward = 0.0
        
        # Initialize the metric structure
        self.episode_metrics = self._init_metrics()
    # Initialize metrics
    def _init_metrics(self, info=None):
        # These are the key metrics we will track during each episode
        # Default values at at the start of an episode
        p_health = info.get('health', 176) if info else 176
        o_health = info.get('enemy_health', 176) if info else 176
        initial_score = info.get('score', 0) if info else 0
        
        # These are default starting x positions
        p_pos = info.get('player_pos', 205) if info else 205
        o_pos = info.get('enemy_pos', 307) if info else 307

        return {
            'player_health': p_health,
            'opponent_health': o_health,
            'previous_score': initial_score,
            'player_pos': p_pos,
            'opponent_pos': o_pos
        }
    # Restart episode metrics after end of episode
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode metrics
        self.episode_metrics = self._init_metrics(info)
        self.previous_action = None
        self.episode_custom_reward = 0.0
        
        return obs, info

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Extract key info
            p_health = info.get('health', 0)
            o_health = info.get('enemy_health', 0)
            current_score = info.get('score', 0)
            p_pos = info.get('player_pos', 205)
            o_pos = info.get('enemy_pos', 307)

            custom_reward = 0.0

            # BRUTE FORCE PERSONA
            # +Damage Dealt Reward
            # -Damage Taken Penalty
            # +Score-based Bonus
            # -Inactivity Penalty
            if self.persona == "brute_force":
                #  Damage Dealt Reward 
                damage_dealt = self.episode_metrics['opponent_health'] - o_health
                if damage_dealt > 0:
                    custom_reward += damage_dealt * 1.0
                    # Bonus for landing a hit
                    custom_reward += 10.0
                
                # Damage Taken Penalty 
                damage_taken = self.episode_metrics['player_health'] - p_health
                if damage_taken > 0:
                    custom_reward -= damage_taken * 0.5

                # Small penalty for completely inactive play
                # (Model would keep finding local optimum of doing nothing)
                if p_health == self.episode_metrics['player_health'] and o_health == self.episode_metrics['opponent_health']:
                    custom_reward -= 0.1
                
                # Score-based Bonus
                # To encourage combos and harder hitting moves 
                score_gained = current_score - self.episode_metrics['previous_score']
                if score_gained > 0:
                    custom_reward += score_gained * 0.001

            # SURVIVAL PERSONA
            # +Consistent Survival Reward
            # -Damage Taken Penalty
            # +Distance Maintenance Reward
            if self.persona == "survival":
                custom_reward += 0.1 # consistent reward for staying alive
                # Damage Taken Penalty 
                damage_taken = self.episode_metrics['player_health'] - p_health
                if damage_taken > 0:
                    custom_reward -= damage_taken * 0.5
                distance = abs(p_pos - o_pos)
                # Reward for maintaining distance
                if distance > self.previous_distance:
                    custom_reward += 1.0    
            #  End of Episode
            if terminated or truncated:
                if o_health <= 0 and p_health > 0:
                    custom_reward += 100.0  # Win bonus
                # Only apply loss penalty for brute_force persona
                elif p_health <= 0 and self.persona == "brute_force":
                    custom_reward -= 75.0  # Loss penalty
                
                # Add custom reward summary to info
                info['custom_reward'] = self.episode_custom_reward + custom_reward
            
            # Track cumulative custom reward
            self.episode_custom_reward += custom_reward
            
            # --- Update metrics ---
            self.episode_metrics.update({
                'player_health': p_health,
                'opponent_health': o_health,
                'previous_score': current_score,
                'previous_pos': p_pos,
                'opponent_pos': o_pos
            })
            self.previous_action = action

            return obs, reward + custom_reward, terminated, truncated, info

        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            raise e


#  Environment Creation Function 
def make_env(env_id: str, seed: int = 0, render: bool = False) -> callable:

    def _init():
        # Parse the ENV_ID: GAME:STATE:PERSONA
        game, state, persona = env_id.split(":")
        render_mode = 'human' if render else None
        # 1. Create the base environment
        env = StreetFighterEnv(game=game, state=state, persona=persona, player=1, render_mode=render_mode)
        
        # 2. Apply Monitor wrapper
        env = Monitor(env, filename=None, allow_early_resets=True)
        
        # 3. Apply frame preprocessing (for quicker training)
        env = WarpFrame(env, width=84, height=84)
        
        # Set seed
        env.reset(seed=seed)
        
        return env
    
    return _init


# Main Training Function 
def main():
    parser = argparse.ArgumentParser(description="DRL for Street Fighter II")
    
    # Model and training args
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"], help="RL algorithm to use.")
    parser.add_argument("--persona", type=str, default="brute_force", choices=["brute_force", "survival"], help="Reward shaping persona.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.") # -> speed up training for each env (modify as needed based on your hardware)
    
    # Environment args
    parser.add_argument("--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis", help="Gym Retro game ID.")
    parser.add_argument("--state", type=str, default="Champion.Level1.RyuVsGuile", help="Gym Retro state ID.")
    
    # Output args
    parser.add_argument("--model-dir", type=str, default="./sf2_models", help="Directory to save models.")
    parser.add_argument("--log-dir", type=str, default="./sf2_logs", help="Directory for TensorBoard logs.")
    
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # --- Environment Setup ---
    env_id = f"{args.game}:{args.state}:{args.persona}"
    n_stack = 4

    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Persona: {args.persona.upper()}")
    print(f"Game: {args.game}")
    print(f"State: {args.state}")
    print(f"Parallel Envs: {args.n_envs}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Frame Stack: {n_stack}")
    print(f"{'='*70}\n")

    # Create vectorized environments
    # Trains the model on multiple instances of the environment in parallel
    env_fns = [make_env(env_id, args.seed + i, render=True) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=n_stack)
    # Normalize rewards -> stabilize training
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    # Model Configs
    if args.algo == "ppo":
        model_class = PPO
        hyperparams = {
            "n_steps": 1024,
            "batch_size": 128,
            "gamma": 0.99,
            "learning_rate": 2.5e-4,
            "ent_coef": 0.05,
            "clip_range": 0.1,
        }
    elif args.algo == "a2c":
        model_class = A2C
        hyperparams = {
            "n_steps": 5,
            "gamma": 0.99,
            "learning_rate": 7e-4,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Initialize the model
    model = model_class(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, f"{args.algo}_{args.persona}"),
        seed=args.seed,
        **hyperparams
    )


    # Train
    print(f"Starting training...\n")
    model.learn(
        total_timesteps=args.timesteps, 
        progress_bar=True,
        
    )
    
    # Save model
    model_name = f"sf2_{args.algo}_{args.persona}_{args.timesteps // 1000}k"
    model_path = os.path.join(args.model_dir, model_name)
    model.save(model_path)

    env.save(os.path.join(args.model_dir, f"{model_name}_vecnormalize.pkl"))
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved: {model_path}.zip")
    
    env.close()

if __name__ == "__main__":
    main()
