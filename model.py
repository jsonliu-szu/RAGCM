from abc import ABC

import torch
import torch.nn as nn
from torchvision import models


# RA-Net
class DecoderBlock(nn.Module, ABC):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class up_(nn.Module, ABC):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(up_, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class eca(nn.Module, ABC):
    def __init__(self, k_size=3):
        super(eca, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        z = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(z).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SAE(nn.Module, ABC):
    def __init__(self, in_dim):
        super(SAE, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        return attention


class RPA(nn.Module, ABC):
    def __init__(self, in_ch, o_size, p_size):
        super(RPA, self).__init__()

        self.in_channel = in_ch
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Sequential(nn.Conv2d(in_ch // 4 * 3, in_ch, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True))
        self.pam = SAE(in_ch // 4)
        self.value_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 4, kernel_size=1)
        self.RCE = nn.ModuleList([self._make_rce(in_ch, o_size, size) for size in p_size])
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _make_rce(in_channel, o_size, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channel, in_channel // 4, 1),
            nn.UpsamplingBilinear2d(size=(o_size, o_size))
        )

    def forward(self, x):
        n, c, height, width = x.size()
        c = c // 4
        priors = [self.pam(stage(x)) for stage in self.RCE]
        proj_value = self.value_conv(x).view(n, -1, width * height)
        out = []
        for attention in priors:
            temp = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out.append(temp.view(n, c, height, width))
        out = self.fusion(torch.cat(out, 1))
        out = self.gamma * out + x
        return out


class APF(nn.Module, ABC):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF, self).__init__()
        self.out_ch = out_ch
        self.up2 = up_(in_ch2, out_ch, scale_factor=2)
        self.up3 = up_(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma * out + x4
        return out


class RA_Net(nn.Module, ABC):
    def __init__(self, n_channels=3, n_classes=1):
        super(RA_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.RPA = RPA(filters[3], 16, [3, 7, 11])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

        self.APF = APF(32, 64, 64, 32)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.RPA(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)

        out = self.APF(out, d1, d2)

        out = self.finalconv3(out)
        return torch.sigmoid(out)
