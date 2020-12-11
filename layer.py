from abc import ABC
import torch
import torch.nn as nn


class up(nn.Module, ABC):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(up, self).__init__()
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
            nn.Conv2d(in_channel,in_channel//4,1),
            nn.UpsamplingBilinear2d(size=(o_size,o_size))
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
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out
