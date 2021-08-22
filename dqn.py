import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

hidden_layer = 16

class DQN(nn.Module):
    

    def __init__(self, actions, device):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        self.device = device
        self.input = nn.Linear(10, hidden_layer)
        self.fc = nn.Linear(hidden_layer, hidden_layer)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_layer, actions)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, device):
        x = x.to(device)
        x = F.relu(self.input(x))
        x = F.relu(self.dropout(self.fc(x)))
        x = F.relu(self.dropout(self.fc(x)))
        x = F.relu(self.dropout(self.fc(x)))
        return self.output(x)
