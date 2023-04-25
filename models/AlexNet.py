import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, classes=2, dropout_p=0.5):
        super().__init__()
        self.classes = classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=192),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=384),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),  # ADDED IN BATCHNORM
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1)
        x = self.head(x)
        return x