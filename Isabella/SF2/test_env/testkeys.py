#used to test tracking of player and opponent x positions in Street Fighter II environment with found ram variables
import retro
import time

# Load Street Fighter II environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

# Reset to starting state
obs = env.reset()

# Run until episode ends
done = False
while not done:
    # Take a random action (for testing)
    obs, rew, done, trunc, info = env.step(env.action_space.sample())
    
    # Print tracked variables if available
    if 'player_pos' in info and 'enemy_pos' in info:
        print(f"Player X Position: {info['player_pos']}, Opponent X Position: {info['enemy_pos']}")

    # Slow down printing so you can see it update
    time.sleep(0.05)

env.close()
