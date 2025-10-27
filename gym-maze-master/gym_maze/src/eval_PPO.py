##BOILERPLATE classcode

##BOILERPLATE classcode
#i think collecting:
#episode 
#reward
#steps , timesteps taken 
#success (if reached goal
#truncated if max steps reached)
#coverage the percent of maze cells, unique only 

#command 
#python eval_PPO.py --model_path ./models/ppo_model

import argparse, os, csv
import numpy as np
import time #keeping incase need to make delay
from stable_baselines3 import PPO

#TODO: recheck this!!! because flappy bird sample imports diferently, wondr if i made mistake
from gym_maze.envs.maze_env import MazeEnv 

def run_episode(model, maze_path, reward_mode="explorer", render=False):
    env = MazeEnv(maze_file=maze_path, reward_design=reward_mode, enable_render=render)
    obs, info = env.reset()
    #done = trunc = False gray not being used
   

    ep_reward = 0.0
    steps = 0
    #clicks = 0  # number of flaps (action==1)
    while True: 
        action, _ = model.predict(obs, deterministic = True) #i think this looping causing of issue of agent going back to previous positions again and again
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        steps += 1
        if terminated or truncated:
             break


    # Episode-level metrics from env info
    success = int(info.get("goal_reached", False))
    truncated_flag = int(truncated)
    visited = info.get("visited_cells", 0)
    total = info.get("total_cells",1)
    coverage = float(visited)/float(total) if total>0 else 0.0

    env.close()

    return {
        "reward": float(ep_reward),
        "steps": steps,
        "success": success,
        #"crashed": int(done and not trunc),
        "truncated": int(truncated_flag),
        "coverage": coverage
    }

#TODO: fix this so many issues with file saving and directories.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="./models/ppo_explorer") #might need to fix directory
    p.add_argument("--maze_path", type=str, default="../envs/maze_samples/maze2d_5x5.npy")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="explorer", choices=["explorer", "survivor"]) #should i keep both option, check later
    #p.add_argument("--csv_out", type=str, default="./logs/eval_metrics.csv")
    p.add_argument("--csv_out", type=str, default=None)
    #p.add_argument("render, type=int, default =0, 
    #p.add_argument("
    args = p.parse_args()

#TODO: add the file saving code 
    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

   #added to match a2c
    if args.csv_out is None:
      model_name = os.path.basename(args.model_path)
      args.csv_out = f"./logs/eval_{model_name}_{args.reward_mode}.csv"


    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    model = PPO.load(args.model_path)

    rows = []

    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, maze_path= args.maze_path,reward_mode=args.reward_mode, render=bool(args.render))
        metrics["episode"] = ep
        rows.append(metrics)

        # Summary
    rewards = [r["reward"] for r in rows]
    successes  = [r["success"] for r in rows]
    coverages  = [r["coverage"] for r in rows]
    #mean_clicks = float(np.mean([r["clicks"] for r in rows]))
    #scrash_rate  = float(np.mean([r["crashed"] for r in rows]))

    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"Mean coverage : {np.mean(coverages):.3f}")
    #print(f"Crash rate: {crash_rate*100:.1f}%")

    # Per-episode CSV
    fieldnames = ["episode","reward","steps" , "success","truncated","coverage"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved metrics to {args.csv_out}")

if __name__ == "__main__":
    main()



'''
SO MANY ISSUES
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        ep_reward += reward
        steps += 1
        #done = trunc or env.done 
        if render:
            env.render()
'''