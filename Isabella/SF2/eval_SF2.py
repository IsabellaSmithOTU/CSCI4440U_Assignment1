import argparse, os, csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
import retro
from stable_baselines3.common.atari_wrappers import WarpFrame
from collections import deque
# Evaluation script for Street Fighter II agent

# Wrapper to stack frames
class FrameStackWrapper(gym.Wrapper):

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
    # reset the environment  
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    # Step the environment (nothing to really step/train here, just stack frames)
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info
    # Get stacked observation
    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=0).squeeze(-1)
# Calculate shaped reward
def calculate_shaped_reward(initial_info, final_info, persona, steps):
    # Get initial and final health values
    initial_p_health = initial_info.get('health_1', 176)
    initial_o_health = initial_info.get('health_2', 176)
    final_p_health = final_info.get('health_1', 0)
    final_o_health = final_info.get('health_2', 0)
    
    total_reward = 0.0
    
    # Damage dealt/taken rewards
    damage_dealt = initial_o_health - final_o_health
    if damage_dealt > 0:
        total_reward += damage_dealt * 1.0
        
    damage_taken = initial_p_health - final_p_health
    if damage_taken > 0:
        total_reward -= damage_taken * 1.5
    
    # Win/loss rewards
    final_matches_won = final_info.get('matches_won', 0)
    final_enemy_matches_won = final_info.get('enemy_matches_won', 0)
    
    if final_matches_won > final_enemy_matches_won:
        total_reward += 100.0  # Win bonus
    elif final_enemy_matches_won > final_matches_won:
        total_reward -= 50.0   # Loss penalty
    
    # Persona-specific penalties (simplified)
    if persona == "brute_force":
        total_reward -= steps * 0.015  # Time penalty
    
    return total_reward

def run_episode(model, game, state, persona, render=False):
    # Create environment with same preprocessing as training
    env = retro.make(
        game=game, 
        state=state, 
        render_mode="human" if render else None,
        use_restricted_actions=retro.Actions.FILTERED
    )
    
    # Apply the same preprocessing wrappers used during training
    env = WarpFrame(env, width=84, height=84)
    env = FrameStackWrapper(env, k=4)
    
    obs, initial_info = env.reset()
    done = False
    steps = 0
    
    # Track match wins/losses
    prev_matches_won = 0
    prev_enemy_matches_won = 0
    match_wins = 0
    match_losses = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(action)
        steps += 1
        
        # Check for match completions
        current_matches_won = info.get('matches_won', 0)
        current_enemy_matches_won = info.get('enemy_matches_won', 0)
        
        # If player won a match
        if current_matches_won > prev_matches_won:
            match_wins += 1
            prev_matches_won = current_matches_won
            
        # If enemy won a match  
        if current_enemy_matches_won > prev_enemy_matches_won:
            match_losses += 1
            prev_enemy_matches_won = current_enemy_matches_won

    final_info = info
    # Calculate shaped reward based on final game state
    shaped_reward = calculate_shaped_reward(initial_info, final_info, persona, steps)
    
    # Final win/loss for the entire episode
    final_matches_won = final_info.get('matches_won', 0)
    final_enemy_matches_won = final_info.get('enemy_matches_won', 0)
    
    if final_matches_won > final_enemy_matches_won:
        episode_win_status = "Win"
        episode_is_win = 1
    elif final_enemy_matches_won > final_matches_won:
        episode_win_status = "Loss" 
        episode_is_win = 0
    else:
        episode_win_status = "Draw"
        episode_is_win = 0

    env.close()
    return {
        "reward": float(shaped_reward),
        "steps": steps,
        "matches_won": final_matches_won,
        "enemy_matches_won": final_enemy_matches_won,
        "match_wins": match_wins,
        "match_losses": match_losses,
        "total_matches": match_wins + match_losses,
        "match_win_rate": match_wins / max(1, match_wins + match_losses),
        "episode_win_status": episode_win_status,
        "episode_is_win": episode_is_win,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis")
    p.add_argument("--state", type=str, default="Champion.Level1.RyuVsGuile")
    p.add_argument("--persona", type=str, default="brute_force")
    p.add_argument("--episodes", type=int, default=10) 
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--csv_out", type=str, default="logs/sf2_eval_metrics.csv")
    args = p.parse_args()

    # Model loading
    model_path = args.model_path
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Determine algorithm
    if "ppo" in args.model_path.lower():
        model = PPO.load(model_path)
        algo = "PPO"
    else:
        model = A2C.load(model_path)
        algo = "A2C"

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    # Run episodes
    rows = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, args.game, args.state, args.persona, render=bool(args.render))
        metrics["episode"] = ep
        metrics["algo"] = algo
        metrics["persona"] = args.persona
        rows.append(metrics)
        print(f"Episode {ep}: Reward={metrics['reward']:.1f}, Matches {metrics['match_wins']}-{metrics['match_losses']} (Win Rate: {metrics['match_win_rate']*100:.1f}%)")

    # Summary statistics
    mean_reward = float(np.mean([r["reward"] for r in rows]))
    std_reward = float(np.std([r["reward"] for r in rows]))
    episode_win_rate = float(np.mean([r["episode_is_win"] for r in rows]))
    match_win_rate = float(np.mean([r["match_win_rate"] for r in rows]))
    mean_steps = float(np.mean([r["steps"] for r in rows]))
    total_matches = sum([r["total_matches"] for r in rows])
    total_match_wins = sum([r["match_wins"] for r in rows])

    print(f"\nEpisodes: {len(rows)}")
    print(f"Mean shaped reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Episode win rate: {episode_win_rate*100:.1f}%")
    print(f"Match win rate: {match_win_rate*100:.1f}% ({total_match_wins}/{total_matches} matches won)")
    print(f"Mean steps: {mean_steps:.1f}")

    # Save CSV
    fieldnames = ["episode", "algo", "persona", "reward", "steps", "matches_won", "enemy_matches_won", 
                  "match_wins", "match_losses", "total_matches", "match_win_rate", "episode_win_status", "episode_is_win"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved metrics to {args.csv_out}")

if __name__ == "__main__":
    main()