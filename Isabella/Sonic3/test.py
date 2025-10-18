#starter code for now
import retro
import time
import cv2
import numpy as np

# initialize 
env = retro.make(
    game='SonicTheHedgehog2-Genesis',
)
obs = env.reset()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sonic_playthrough.avi', fourcc, 60.0, (obs.shape[1], obs.shape[0]))


try:
    for step in range(1000):  
        action = env.action_space.sample() 
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, info, _ = result  # discard extra value

        env.render()  # display

        if done:
            obs = env.reset()  # if the game over

        time.sleep(0.01)
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        

finally:
    env.close() 
