import torch

from IPython import display
from custom_env_tutorial import ChopperScape
from dqn import DQN
from get_copter_pos import get_screen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ChopperScape()
obs = env.reset()

init_screen = get_screen(env)
print('Init screen shape', init_screen.shape)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

model = DQN(screen_height, screen_width, n_actions).to(device)


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Render the game
    env.render()
    for item in env.elements:
        if type(item).__name__ == 'Chopper':
            print('Position: ', item.get_position())
            print('Lifes: ', item.lives)
    
    if done == True:
        break

env.close()