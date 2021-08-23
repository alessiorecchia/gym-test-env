import torch
import math
import random

from IPython import display
from custom_env_tutorial import ChopperScape
from dqn import DQN
from get_copter_pos import get_screen


GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

SAVING_PATH = './model/test_dqn'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ChopperScape()
obs = env.reset()

init_screen = get_screen(env)
print('Init screen shape', init_screen.shape)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

model = DQN(screen_height, screen_width, n_actions).to(device)
model.load_state_dict(torch.load(SAVING_PATH))
model.eval()

last_screen = get_screen(env)
current_screen = get_screen(env)
state = current_screen - last_screen

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


while True:
    # Take a random action
    action = model(state).max(1)[1].view(1, 1)
    print(f'Action: {action}', end="")
    print('\r', end='')
    obs, reward, done, info = env.step(action.item())
    
    # Render the game
    env.render()
    # for item in env.elements:
    #     if type(item).__name__ == 'Chopper':
    #         print('Position: ', item.get_position())
    #         print('Lifes: ', item.lives)
    last_screen = current_screen
    current_screen = get_screen(env)
    state = current_screen - last_screen
    

    
    if done == True:
        break

env.close()