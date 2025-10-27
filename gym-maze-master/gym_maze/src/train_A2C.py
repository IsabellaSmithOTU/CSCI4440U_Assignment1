#commands 

#TRAINING
#python train_ppo.py --reward_mode survivor
#python train_a2c.py --reward_mode explorer --timesteps 5000
#online says
#TESTING

#RUNNING


#using class sample as boilerplate for training
#https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
import time
import argparse
import os
#import csv #for csv files
import gymnasium as gym #recheck if up to date
#from stable_baselines3 import PPO
from stable_baselines3 import A2C
#reconfirm if keep these during debugging
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

#reward mode: survivor, explorer DONE 

#https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
#TODO: Metrics task , im gonna make same for A2C and ppo
#measuring 
#ep reward
#ep len
#plots to compare A2C vs PPO

#some time?






#from flappy_env import FlappyBirdEnv
from gym_maze.envs.maze_env import MazeEnv #reusing oss environment

#using similar concept to class example
#def make_env(render_mode=None, reward_mode="survival", seed=7):
    #env = FlappyBirdEnv(render_mode=render_mode, reward_mode=reward_mode, seed=seed)
    #env = Monitor(env)
   # return env
def maze_env(maze_path: str, seed: int=7, reward_design: str="explorer"): #spelling error

    #pass value of reward design
    env = MazeEnv(maze_file=maze_path, reward_design=reward_design, enable_render=False) #TODO added, debug again
    env.reset(seed=seed)
    env = Monitor(env) #logginh rewards, lengths auto
    return env


###METRICS###
'''
def save_csv(env, csv_path): #csv file saving func
    #monitor data from stablebaselines directly
    results = env.get_episode_lengths(), env.get_episode_rewards(), env.get_episode_times()

    '''
def main():
    parser = argparse.ArgumentParser()
    #TODO modify these relevant to my game
    parser.add_argument("--maze_path", type=str, default="../envs/maze_samples/maze2d_5x5.npy") #fix file directory, I want to see if I can have all the mazes ??? Time permitting
    parser.add_argument("--timesteps", type=int, default=200_000) #TODO: change this value for longer training
    parser.add_argument("--seed", type=int, default=7) #keep from class example
    parser.add_argument("--logdir", type=str, default="./logs") #recheck if modify (document review)
    parser.add_argument("--modeldir", type=str, default="./models")
   # parser.add_argument("--reward_mode", type=str, default="dense") BRING BACK LATER
    parser.add_argument("--reward_mode", type=str, default="explorer")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

   # env = make_env(reward_mode=args.reward_mode, seed=args.seed)
    #env = make_env(seed=args.seed) trial for debuging

#a2c dont work suddenly??
    #using to help resolve https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
    env = maze_env(args.maze_path, seed=args.seed, reward_design=args.reward_mode)  #debug? spelling error
    model = A2C(
        policy="MlpPolicy", #want to try different policy time permitting
        env=env, #i can keep
        verbose=1, #i can keep
        tensorboard_log=args.logdir, #i can keep
        seed=args.seed, #i can keep
        #batch_size=256, n_steps in documentation 
        #n_steps = 256, #TODO check the results after, im thinking maybe remove this if not needed, but then idk if test will be fair?
        n_steps = 5, #online references say https://stable-baselines.readthedocs.io/en/master/modules/a2c.html
        gamma=0.99, #i can keep
        gae_lambda=0.95, #i can keep
        #n_epochs=10, I CANT FIND ALTERNATIVE FOR FAIR COMPARISON!
        learning_rate=3e-4,
        )
       # clip_range=0.2, MAYBE THIS IS ALTERNATIVE? max_grad_norm 
    

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])  # inspect later in TenserB
    model.set_logger(new_logger)

#check for package issues?
    model.learn(total_timesteps=args.timesteps, progress_bar=False) #model training, import error testing with = false

#modifying flappy bird class code 
  #  save_name = f"ppo_flappy_{args.reward_mode}"
  #  path = os.path.join(args.modeldir, save_name)
  #  model.save(path)
  #  print(f"Saved model to {path}")
    save_path = os.path.join(args.modeldir, f"a2c_{args.reward_mode}") #1st model
    model.save(save_path)
   # print("Saved mo")

    #recheck why it was path and not save_path? error in class code?
    print(f"Saved model to {save_path}") #flappy bird used this, TODO figure out to remove the yellow underline



   


if __name__ == "__main__":
    main()

    #for presentation, make sure all the files in correct folder structures and test after coombining, otheriwse teminal errors
    #note A2C is performing better that ppo, time permitting change the time i think



    '''
        #redering, game screen was black
    render_env = MazeEnv(maze_file=args.maze_path, enable_render=True, reward_design=args.reward_mode) #adding reward design, new training TODO
    obs, info = render_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated
       # render_env.close() #testing if game opens
        render_env.render() #black screen keep closing, trying to update game window DEBUG

       # TODO LATER WAY TOO FAST RENDERING, FIX 
        time.sleep(0.05) #incse agent is moving too fast

    render_env.close() #wrongly placed?
    input("Press Enter to exit") #temp debug for issue with window not rendering w/shorter timesteps


#print out rewards and stuff later (Check documen reqs)
    '''