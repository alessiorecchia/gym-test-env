from math import e
import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import torchvision.transforms as T
import collections

from torch.optim.lr_scheduler import StepLR
from PIL import Image

from custom_env_tutorial import ChopperScape

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
UPDATE_REPEATS = 50
MIN_EPISODES = 1
EPISODES = 10
MEMORY = 10000
SAVING_PATH = './model/test_dqn'

resize = T.Compose([T.ToPILImage(),
                    # T.Resize((40, 40), interpolation=Image.CUBIC),
                    T.Resize(80, interpolation=Image.CUBIC),
                    T.ToTensor()])

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    # def __init__(self, action_dim):
    #     super(QNetworkCNN, self).__init__()

    #     self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
    #     self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
    #     self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    #     self.fc_1 = nn.Linear(8960, 512)
    #     self.fc_2 = nn.Linear(512, action_dim)

    # def forward(self, inp):
    #     inp = inp.view((1, 3, 210, 160))
    #     x1 = F.relu(self.conv_1(inp))
    #     x1 = F.relu(self.conv_2(x1))
    #     x1 = F.relu(self.conv_3(x1))
    #     x1 = torch.flatten(x1, 1)
    #     x1 = F.leaky_relu(self.fc_1(x1))
    #     x1 = self.fc_2(x1)

    #     return x1

    def __init__(self, h, w, outputs):
        super(QNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))), kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))), kernel_size=3, stride=1)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))
        # x = self.head(x.view(x.size(0), -1))
        # return F.softmax(x, dim=1)


"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_action(model, env, state, eps):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform/repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


def main(env_name, h, w, actions, gamma=GAMMA, lr=1e-3, min_episodes=MIN_EPISODES, eps=1, eps_decay=EPS_DECAY, eps_min=EPS_END, update_step=TARGET_UPDATE, batch_size=BATCH_SIZE, update_repeats=UPDATE_REPEATS,
         num_episodes=EPISODES, seed=42, max_memory_size=MEMORY, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, cnn=False, horizon=np.inf, render=True, render_step=50):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    env = env_name
    torch.manual_seed(seed)
    # env.seed(seed)
    # h, w, outputs
    if cnn:
        Q_1 = QNetworkCNN(h, w, actions).to(device)
        Q_2 = QNetworkCNN(h, w, actions).to(device)
    else:
        Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
        Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        # display the performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

        state = env.reset()
        memory.state.append(state)

        done = False
        i = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state, eps)
            _, reward, done, _ = env.step(action)
            state = get_screen(env)

            if i > horizon:
                done = True

            # render the environment if render == True
            if render and episode % render_step == 0:
                env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps*eps_decay, eps_min)

    # return Q_1, performance
    torch.save(Q_1.state_dict(), SAVING_PATH)


env = ChopperScape()
env.reset()

init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

if __name__ == '__main__':
    main(env, screen_height, screen_width, n_actions, cnn=True, render=False)
