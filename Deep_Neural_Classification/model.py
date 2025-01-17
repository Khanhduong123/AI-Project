import torch.nn as nn

_LAYER_CONFIG= {16: [2,2,3,3,3]}

class VGG(nn.Module):
    def __init__(self) -> None:
        super(VGG, self).__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.features = self._make_layers(_LAYER_CONFIG[16])
        self.classifier = nn.Linear(512, 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        


    def _make_layers(self, definitions):
        layers = []
        in_channels = 3

        for num_layers, channels in zip(definitions, self.channels):
            for _ in range(num_layers):
                """
                layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(channels))
                layers.append(nn.ReLU(inplace=True))
                """
                layers += [nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(channels),
                           nn.ReLU(inplace=True)]
               
                in_channels = channels
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)