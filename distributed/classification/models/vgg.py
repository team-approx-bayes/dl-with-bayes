'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torchsso.utils.accumulator import TensorAccumulator


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, vgg_name='VGG19'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG19(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)), inplace=True)
        h = F.relu(self.bn1_2(self.conv1_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn2_1(self.conv2_1(h)), inplace=True)
        h = F.relu(self.bn2_2(self.conv2_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn3_1(self.conv3_1(h)), inplace=True)
        h = F.relu(self.bn3_2(self.conv3_2(h)), inplace=True)
        h = F.relu(self.bn3_3(self.conv3_3(h)), inplace=True)
        h = F.relu(self.bn3_4(self.conv3_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn4_1(self.conv4_1(h)), inplace=True)
        h = F.relu(self.bn4_2(self.conv4_2(h)), inplace=True)
        h = F.relu(self.bn4_3(self.conv4_3(h)), inplace=True)
        h = F.relu(self.bn4_4(self.conv4_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn5_1(self.conv5_1(h)), inplace=True)
        h = F.relu(self.bn5_2(self.conv5_2(h)), inplace=True)
        h = F.relu(self.bn5_3(self.conv5_3(h)), inplace=True)
        h = F.relu(self.bn5_4(self.conv5_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out


class VGG19MCDropout(VGG19):

    def __init__(self, num_classes=10, dropout_ratio=0.1, val_mc=10):
        super(VGG19MCDropout, self).__init__(num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        p = self.dropout_ratio
        h = F.relu(self.bn1_1(F.dropout(self.conv1_1(x), p)), inplace=True)
        h = F.relu(self.bn1_2(F.dropout(self.conv1_2(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn2_1(F.dropout(self.conv2_1(h), p)), inplace=True)
        h = F.relu(self.bn2_2(F.dropout(self.conv2_2(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn3_1(F.dropout(self.conv3_1(h), p)), inplace=True)
        h = F.relu(self.bn3_2(F.dropout(self.conv3_2(h), p)), inplace=True)
        h = F.relu(self.bn3_3(F.dropout(self.conv3_3(h), p)), inplace=True)
        h = F.relu(self.bn3_4(F.dropout(self.conv3_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn4_1(F.dropout(self.conv4_1(h), p)), inplace=True)
        h = F.relu(self.bn4_2(F.dropout(self.conv4_2(h), p)), inplace=True)
        h = F.relu(self.bn4_3(F.dropout(self.conv4_3(h), p)), inplace=True)
        h = F.relu(self.bn4_4(F.dropout(self.conv4_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn5_1(F.dropout(self.conv5_1(h), p)), inplace=True)
        h = F.relu(self.bn5_2(F.dropout(self.conv5_2(h), p)), inplace=True)
        h = F.relu(self.bn5_3(F.dropout(self.conv5_3(h), p)), inplace=True)
        h = F.relu(self.bn5_4(F.dropout(self.conv5_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        out = F.dropout(self.fc(h), p)
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
