#Used to visually identify which buttons correspond to which actions in Street Fighter II

import retro
import numpy as np
import time

def visual_button_test():
    env = retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis",
        state="Champion.Level1.RyuVsGuile", 
        render_mode='human'
    )
    
    
    obs, info = env.reset()
    no_op = np.zeros(12, dtype=bool) # -> 12 buttons on the Genesis controller

def identify_attack_buttons():
    """Quick test to identify which buttons are attacks"""
    env = retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis",
        state="Champion.Level1.RyuVsGuile",
        render_mode='human'
    )
    
    obs, info = env.reset()
    no_op = np.zeros(12, dtype=bool)
    
    # Common attack button indices in fighting games
    likely_attack_buttons = [0,0,0,0,0,0,0,3,3,3,3,3,6,6,6,6,6] # Example indices for punches and kicks
    # I put a bunch of duplicates to give more time to see the effect of each button
    # Here's what I think they are:
    # 0 = kick ???
    # 1 = kick
    # 2 = idle
    # 3 = idle?
    # 4 = up
    # 5 = down
    # 6 = dodge
    # 7 = right
    # 8 = kick
    # 9 = punch
    # 10 = punch
    # 11 = punch
    for button_idx in likely_attack_buttons:
        print(f"\nTesting button {button_idx} ")
      
        test_action = no_op.copy()
        test_action[button_idx] = True
        
        print("Testing attack...")
        for _ in range(15):  # Press attack for 15 frames
            obs, reward, terminated, truncated, info = env.step(test_action)
            time.sleep(0.05)
            if terminated:
                obs, info = env.reset()
                break
        
        # Return to neutral
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(no_op)
            time.sleep(0.03)
        
        input("Press Enter to continue...")
    
    env.close()

if __name__ == "__main__":

    identify_attack_buttons()