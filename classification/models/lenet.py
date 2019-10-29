import torch.nn as nn
import torch.nn.functional as F
from torchsso.utils.accumulator import TensorAccumulator


class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet5MCDropout(LeNet5):

    def __init__(self, num_classes=10, dropout_ratio=0.1, val_mc=10):
        super(LeNet5MCDropout, self).__init__(num_classes=num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        p = self.dropout_ratio
        out = F.relu(F.dropout(self.conv1(x), p))
        out = F.max_pool2d(out, 2)
        out = F.relu(F.dropout(self.conv2(out), p))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(F.dropout(self.fc1(out), p))
        out = F.relu(F.dropout(self.fc2(out), p))
        out = F.dropout(self.fc2(out), p)
        return out

    def mc_prediction(self, x):

        acc_prob = TensorAccumulator()
        m = self.val_mc

        for _ in range(m):
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1/m)

        prob = acc_prob.get()

        return prob


class LeNet5BatchNorm(nn.Module):
    def __init__(self, num_classes=10, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6, affine=affine)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=affine)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=affine)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn3(self.fc1(out)))
        out = F.relu(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out
