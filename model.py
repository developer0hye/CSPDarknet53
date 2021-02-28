import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


ACTIVATIONS = {
    'mish': Mish(),
    'leaky': nn.LeakyReLU(negative_slope=0.1),
    'linear': nn.Identity()
}


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv(x)


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear'):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            Conv(in_channels, hidden_channels, 1),
            Conv(hidden_channels, out_channels, 3)
        )

        self.activation = ACTIVATIONS[residual_activation]

    def forward(self, x):
        return self.activation(x+self.block(x))


class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = Conv(out_channels, out_channels, 1)
        self.split_conv1 = Conv(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Conv(out_channels, out_channels, 1)
        )

        self.concat_conv = Conv(out_channels*2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x


class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = Conv(out_channels, out_channels//2, 1)
        self.split_conv1 = Conv(out_channels, out_channels//2, 1)

        self.blocks_conv = nn.Sequential(
            *[CSPBlock(out_channels//2, out_channels//2)
                       for _ in range(num_blocks)],
            Conv(out_channels//2, out_channels//2, 1)
        )

        self.concat_conv = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x


class CSPDarknet53(nn.Module):
  def __init__(self, 
               stem_channels=32, 
               feature_channels=[64, 128, 256, 512, 1024], 
               num_features=3,
               num_classes=1000):
    
    super(CSPDarknet53, self).__init__()

    self.stem_conv = Conv(3, stem_channels, 3)

    self.stages = nn.ModuleList([
      CSPFirstStage(stem_channels, feature_channels[0]),
      CSPStage(feature_channels[0], feature_channels[1], 2),
      CSPStage(feature_channels[1], feature_channels[2], 8),
      CSPStage(feature_channels[2], feature_channels[3], 8),
      CSPStage(feature_channels[3], feature_channels[4], 4)
    ])

    self.feature_channels = feature_channels
    self.num_classes = num_classes

    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Conv2d(1024, self.num_classes, 1)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
          
  def extract_features(self, x):
    features = []

    x = self.stem_conv(x)

    x = self.stages[0](x)#//2
    x = self.stages[1](x)#//4
    x8 = self.stages[2](x)#//8
    features.append(x8)

    x16 = self.stages[3](x8)#//16
    features.append(x16)

    x32 = self.stages[4](x16)#//32
    features.append(x32)

    return features

  def forward(self, x):
    features = self.extract_features(x)
    x = self.gap(features[-1])
    x = self.fc(x)
    x = x.flatten(start_dim=1)
    return x

if __name__ == '__main__':
    model = CSPDarknet53(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
