import argparse
import time
from stable_baselines3 import PPO
from gym_maze.envs.maze_env import MazeEnv

def main():
    parser = argparse.ArgumentParser()
    #TODO modify these relevant to my game
    parser.add_argument("--model_path", type=str, default="./models/ppo_explorer") #fix file directory
    parser.add_argument("--maze_path", type=str, default="../envs/maze_samples/maze2d_5x5.npy") #TODO: change this value for longer training
  #  parser.add_argument("--seed", type=int, default=7) #keep from class example
   # parser.add_argument("--logdir", type=str, default="./logs") #recheck if modify (document review)
   # parser.add_argument("--modeldir", type=str, default="./models")
   # parser.add_argument("--reward_mode", type=str, default="dense") BRING BACK LATER
    parser.add_argument("--reward_mode", type=str, default="explorer", choices = ["explorer", "survivor"])# the agent was not moving without this
    args = parser.parse_args()
    
    model = PPO.load(args.model_path)

    #redering, game screen was black
    render_env = MazeEnv(maze_file=args.maze_path, enable_render=True, reward_design=args.reward_mode) #reward_design="explorer") #adding design
    obs, info = render_env.reset()
    time.sleep(0.2) #slowing down window, too fast render
    #done = False

    terminated = truncated = False
    total_reward = 0
    steps = 0

    #os.makedirs(args.logdir, exist_ok=True)
    #os.makedirs(args.modeldir, exist_ok=True)

    print("Rnning vis")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True) #i think this looping causing of issue of agent going back to previous positions again and again
        obs, reward, terminated, truncated, info = render_env.step(action)
        #done = terminated or truncated
        total_reward += reward
        steps +=1
       # render_env.close() #testing if game opens
        render_env.render() #black screen keep closing, trying to update game window DEBUG

       # TODO LATER WAY TOO FAST RENDERING, FIX 
        time.sleep(0.05) #incse agent is moving too fast

    render_env.close() #wrongly placed?


 

#print out rewards and stuff later (Check document reqs)
if __name__ == "__main__":
    main()