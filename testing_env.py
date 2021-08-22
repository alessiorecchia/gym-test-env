from IPython import display
from custom_env_tutorial import ChopperScape

env = ChopperScape()
obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()