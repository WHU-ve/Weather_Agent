import torch
import torch.nn as nn
import torch.nn.functional as F


factor = 2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed3a(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = BasicConv2d(32 // factor, 48 // factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        return torch.cat((x0, x1), 1)


class Mixed4a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(80 // factor, 32 // factor, kernel_size=1, stride=1),
            BasicConv2d(32 // factor, 48 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(80 // factor, 32 // factor, kernel_size=1, stride=1),
            BasicConv2d(32 // factor, 32 // factor, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(32 // factor, 32 // factor, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(32 // factor, 48 // factor, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        return torch.cat((x0, x1), 1)


class Mixed5a(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = BasicConv2d(96 // factor, 96 // factor, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        return torch.cat((x0, x1), 1)


class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(192 // factor, 48 // factor, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(192 // factor, 32 // factor, kernel_size=1, stride=1),
            BasicConv2d(32 // factor, 48 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(192 // factor, 32 // factor, kernel_size=1, stride=1),
            BasicConv2d(32 // factor, 48 // factor, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48 // factor, 48 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192 // factor, 48 // factor, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(192 // factor, 192 // factor, kernel_size=3, stride=1, padding=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(192 // factor, 96 // factor, kernel_size=1, stride=1),
            BasicConv2d(96 // factor, 112 // factor, kernel_size=3, stride=1, padding=1),
            BasicConv2d(112 // factor, 128 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), 1)


class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(512 // factor, 192 // factor, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(512 // factor, 96 // factor, kernel_size=1, stride=1),
            BasicConv2d(96 // factor, 112 // factor, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(112 // factor, 128 // factor, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(512 // factor, 96 // factor, kernel_size=1, stride=1),
            BasicConv2d(96 // factor, 96 // factor, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(96 // factor, 112 // factor, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(112 // factor, 112 // factor, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(112 // factor, 128 // factor, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(512 // factor, 64 // factor, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(512 // factor, 96 // factor, kernel_size=1, stride=1),
            BasicConv2d(96 // factor, 96 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(512 // factor, 128 // factor, kernel_size=1, stride=1),
            BasicConv2d(128 // factor, 128 // factor, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128 // factor, 160 // factor, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(160 // factor, 160 // factor, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), 1)


class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(768 // factor, 128 // factor, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(768 // factor, 192 // factor, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(192 // factor, 128 // factor, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(192 // factor, 128 // factor, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = BasicConv2d(768 // factor, 192 // factor, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(192 // factor, 224 // factor, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(224 // factor, 256 // factor, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(256 // factor, 128 // factor, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(256 // factor, 128 // factor, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(768 // factor, 128 // factor, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        return torch.cat((x0, x1, x2, x3), 1)


class InceptionV4(nn.Module):
    def __init__(self, in_chans=3, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate
        self.features = nn.Sequential(
            BasicConv2d(in_chans, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 32 // factor, kernel_size=3, stride=1, padding=1),
            Mixed3a(),
            Mixed4a(),
            Mixed5a(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            ReductionA(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            ReductionB(),
            InceptionC(),
            InceptionC(),
            InceptionC(),
        )

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class DilationPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilatedConv0 = nn.Conv2d(768 // factor, 384 // factor, kernel_size=3, dilation=1, padding=1)
        self.dilatedConv1 = nn.Conv2d(768 // factor, 384 // factor, kernel_size=3, dilation=2, padding=2)
        self.dilatedConv2 = nn.Conv2d(768 // factor, 384 // factor, kernel_size=3, dilation=4, padding=4)
        self.dilatedConv3 = nn.Conv2d(768 // factor, 384 // factor, kernel_size=3, dilation=8, padding=8)
        self.dilatedConv4 = nn.Conv2d(768 // factor, 384 // factor, kernel_size=3, dilation=16, padding=16)

    def forward(self, x):
        out0 = self.dilatedConv0(x)
        out1 = self.dilatedConv1(x)
        out2 = self.dilatedConv2(x)
        out3 = self.dilatedConv3(x)
        out4 = self.dilatedConv4(x)
        return torch.cat((out0, out1, out2, out3, out4), 1)


class DescriptorT(nn.Module):
    def __init__(self):
        super().__init__()
        self.iv4 = InceptionV4()
        self.dp = DilationPyramid()

    def forward(self, x):
        return self.dp(self.iv4(x))


class PyramidMaxout(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1920 // factor, out_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(1920 // factor, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1920 // factor, out_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(1920 // factor, out_dim, kernel_size=7, padding=3)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out_a = torch.maximum(out1, out3)
        out_b = torch.maximum(out5, out7)
        return torch.maximum(out_a, out_b)


class PyramidSum(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1920 // factor, out_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(1920 // factor, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1920 // factor, out_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(1920 // factor, out_dim, kernel_size=7, padding=3)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out_a = torch.add(out1, out3)
        out_b = torch.add(out5, out7)
        return torch.add(out_a, out_b)


class SnowExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramidMaxout = PyramidMaxout(out_dim=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.pyramidMaxout(x)
        return self.prelu(out)


class AberrationExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramidMaxout = PyramidMaxout(out_dim=3)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.pyramidMaxout(x)
        return self.prelu(out)


class DescriptorR(nn.Module):
    def __init__(self):
        super().__init__()
        self.iv4 = InceptionV4(in_chans=7)
        self.dp = DilationPyramid()

    def forward(self, x):
        return self.dp(self.iv4(x))


class RecoveryR(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramidSum = PyramidSum(out_dim=3)

    def forward(self, x):
        return self.pyramidSum(x)


class DeSnowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.descriptorT = DescriptorT()
        self.snowExtractor = SnowExtractor()
        self.aberrationExtractor = AberrationExtractor()
        self.descriptorR = DescriptorR()
        self.recoveryR = RecoveryR()

    def forward(self, x):
        f_t = self.descriptorT(x)
        z_hat = self.snowExtractor(f_t)
        a = self.aberrationExtractor(f_t)
        y_dash = self.recover(x, z_hat, a)
        f_c = torch.cat((y_dash, z_hat, a), 1)
        f_r = self.descriptorR(f_c)
        r = self.recoveryR(f_r)
        y_hat = torch.add(y_dash, r)
        return y_hat, y_dash, z_hat

    @staticmethod
    def recover(x, z_hat, a):
        mask = z_hat < 1.0
        out = (x - (a * z_hat)) / (1 - z_hat)
        out = out * mask + x * (~mask)
        return out