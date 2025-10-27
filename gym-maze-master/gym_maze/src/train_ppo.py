#using class sample as boilerplate for training and the train_ppo.py
import time
import argparse
import os
import gymnasium as gym #recheck if up to date
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

#from flappy_env import FlappyBirdEnv
from gym_maze.envs.maze_env import MazeEnv #reusing oss environment

#using similar concept to class example
#def make_env(render_mode=None, reward_mode="survival", seed=7):
    #env = FlappyBirdEnv(render_mode=render_mode, reward_mode=reward_mode, seed=seed)
    #env = Monitor(env)
   # return env
def maze_env(maze_path: str, seed: int=7, reward_design: str="explorer"): #spelling error
    env = MazeEnv(maze_file=maze_path, reward_design=reward_design, enable_render=False) #TODO added, debug again, Maybe work? (to avoid black screen)
    env.reset(seed=seed) #added seed again
    env = Monitor(env) #logginh rewards, lengths auto
    return env

def main():
    parser = argparse.ArgumentParser()
    #TODO modify these relevant to my game
    parser.add_argument("--maze_path", type=str, default="../envs/maze_samples/maze2d_5x5.npy") #fix file directory
    parser.add_argument("--timesteps", type=int, default=200_000) #TODO: change this value for longer training
    parser.add_argument("--seed", type=int, default=7) #keep from class example
    parser.add_argument("--logdir", type=str, default="./logs") #recheck if modify (document review)
    parser.add_argument("--modeldir", type=str, default="./models")
   # parser.add_argument("--reward_mode", type=str, default="dense") BRING BACK LATER
    parser.add_argument("--reward_mode", type=str, default="explorer") #the agent was not moving without this
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

   # env = make_env(reward_mode=args.reward_mode, seed=args.seed)
    #env = make_env(seed=args.seed) trial for debuging
    env = maze_env(args.maze_path, seed=args.seed, reward_design=args.reward_mode)  #debug? spelling error
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])  # later will nice to inspect in TB
    model.set_logger(new_logger)

#check for package issues?
    model.learn(total_timesteps=args.timesteps, progress_bar=False) #model training, import error testing with = false

#modifying flappy bird class code 
  #  save_name = f"ppo_flappy_{args.reward_mode}"
  #  path = os.path.join(args.modeldir, save_name)
  #  model.save(path)
  #  print(f"Saved model to {path}")
    save_path = os.path.join(args.modeldir, f"ppo_{args.reward_mode}") #dependent on model and reward_mode
    model.save(save_path)
    #print(f"Saved model to {save_path}") #printing saved path, matching A2C
   # print("Saved mo")

    #recheck why it was path and not save_path? error in class code?
    print(f"Saved model to {save_path}") #flappy bird used this, TODO figure out to remove the yellow underline

   

   


if __name__ == "__main__":
    main()

    #for presentation, make sure all the files in correct folder structures and test after coombining, otheriwse teminal errors



    #REMNDER
    #Accidentally modified ppo file code instead of A2C, TODO: switch the file names and recheck after before submission