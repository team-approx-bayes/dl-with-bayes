'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
from torchsso.utils.accumulator import TensorAccumulator


__all__ = ['alexnet', 'alexnet_mcdropout']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AlexNet2(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNetMCDropout(AlexNet):

    def __init__(self, num_classes=10, dropout_ratio=0.5, val_mc=10):
        super(AlexNetMCDropout, self).__init__(num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        dropout_ratio = self.dropout_ratio
        x = F.relu(F.dropout(self.conv1(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.dropout(self.conv2(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.dropout(self.conv3(x), p=dropout_ratio), inplace=True)
        x = F.relu(F.dropout(self.conv4(x), p=dropout_ratio), inplace=True)
        x = F.relu(F.dropout(self.conv5(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def mc_prediction(self, x):

        acc_prob = TensorAccumulator()
        m = self.val_mc

        for _ in range(m):
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1/m)

        prob = acc_prob.get()

        return prob


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


def alexnet_mcdropout(**kwargs):
    model = AlexNetMCDropout(**kwargs)
    return model

