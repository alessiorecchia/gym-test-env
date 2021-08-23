import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from custom_env_tutorial import ChopperScape

env = ChopperScape()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

resize = T.Compose([T.ToPILImage(),
                    # T.Resize((40, 40), interpolation=Image.CUBIC),
                    T.Resize(80, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_copter_location(env):
    for element in env.elements:
        if type(element).__name__ == 'Chopper':
            return element.get_position()

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # print('\n\n\n##############\n')
    # print('in-function screen var shape: ', screen.shape)
    # print('\n\n\n##############\n')
    # Cart is in the lower half, so strip off the top and bottom of the screen
    # _, screen_height, screen_width = screen.shape
    # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # view_width = int(screen_width * 0.6)
    # copter_location = get_copter_location(env)
    # print('Copter Location: ', copter_location)
    # print('\n\n\n##############\n')
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, copter_location[0]:copter_location[0]+64, copter_location[1]:copter_location[1]+64]
    # print(screen.shape)
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


# env.reset()
# plt.figure()
# plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()