import torch.nn as nn


class MWTCNN(nn.Module):
    def __init__(self):
        super(MWTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 9, padding=4, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 6, padding=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)

        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.classifier(x)

        return x
