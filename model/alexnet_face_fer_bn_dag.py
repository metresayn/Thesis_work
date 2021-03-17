import torch
import torch.nn as nn


class Alexnet_face_fer_bn_dag(nn.Module):

    def __init__(self):
        super(Alexnet_face_fer_bn_dag, self).__init__()
        self.meta = {'mean': [131.09375, 103.88607788085938, 91.47599792480469],
                     'std': [1, 1, 1],
                     'imageSize': [227, 227, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[11, 11], stride=(4, 4))
        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(1, 1), padding=(2, 2), groups=2)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.bn4 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=1, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn7 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=7, bias=True)

    def forward(self, data):
        x1 = self.conv1(data)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.pool2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)
        x1 = self.conv4(x1)
        x1 = self.bn4(x1)
        x1 = self.relu4(x1)
        x1 = self.conv5(x1)
        x1 = self.bn5(x1)
        x1 = self.relu5(x1)
        x1 = self.pool5(x1)
        x1 = self.fc6(x1)
        x1 = self.bn6(x1)
        x1 = self.relu6(x1)
        x1 = self.fc7(x1)
        x1 = self.bn7(x1)
        x1 = self.relu7(x1)
        x1 = x1.view(x1.size(0), -1)
        #prediction = self.fc8(x1)
        return x1