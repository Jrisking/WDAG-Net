from lib.Modules import BasicConv2d, BasicDeConv2d
import torch.nn as nn
from lib.CBAM import SpatialAttention
affine_par = True
import torch
from torch.nn import functional as F
from lib.GatedConv import GatedConv2dWithActivation



class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1))
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        return out

class TFD(nn.Module):
    def __init__(self, in_channels):
        super(TFD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.gatedconv = GatedConv2dWithActivation(in_channels * 2, in_channels, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
                                                   activation=torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feature_map, perior_repeat):
        assert (feature_map.shape == perior_repeat.shape), "feature_map and prior_repeat have different shape"
        uj = perior_repeat
        uj_conv = self.conv(uj)
        uj_1 = uj_conv + uj
        uj_i_feature = torch.cat([uj_1, feature_map], 1)
        uj_2 = uj_1 + self.gatedconv(uj_i_feature) - 3 * uj_conv
        return uj_2


class REU6(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REU6, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 3, mid_channels * 2, kernel_size=3, padding=1),
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.TFD = TFD(in_channels)
        self.out_B = nn.Sequential(
            BasicDeConv2d(in_channels, mid_channels // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.ode1 = ODE(in_channels)
        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )
        self.sa = SpatialAttention()

        # conv block
        self.blockf1 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)  # in_channels * 2, in_channels, kernel_size=3, padding=1
        self.blockf2 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.blockf3 = BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.block = BasicConv2d(in_channels, in_channels, 1)

    def forward(self, x, prior_cam):
        f1 = x
        f2 = x
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear')  # 2,1,12,12->2,1,48,48
        prior_cam_r = prior_cam.expand(-1, x.size()[1], -1, -1)

        f1 = (-1 * (torch.sigmoid(prior_cam)) + 1).expand(-1, x.size()[1], -1, -1).mul(x) + f1
        f1 = self.global_att(f1) * f1   # ablation study

        f2 = self.block(prior_cam_r + f2)

        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1))
        yt = self.sa(yt) * yt
        yt = self.ode1(yt)

        f1 = self.blockf1(torch.cat([f1, yt], dim=1))

        # yt = self.ode2(yt)
        # f2 = self.blockf2(torch.cat([f2, f1], dim=1))
        f2 = f2 + f1

        f2 = self.local_att(f2) * f2  # ablation study
        f2 = self.blockf3(torch.cat([f2, yt], dim=1))

        bound = self.out_B(yt)
        bound = self.edge_enhance(bound)

        y = torch.cat([f1, yt, f2], dim=1)

        y = self.out_y(y)
        y = y + prior_cam
        return y, bound

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out

class REM12(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REM12, self).__init__()
        self.REU_f1 = REU6(in_channels, mid_channels)
        self.REU_f2 = REU6(in_channels, mid_channels)
        self.REU_f3 = REU6(in_channels, mid_channels)
        self.REU_f4 = REU6(in_channels, mid_channels)

    def forward(self, x, prior_0, pic):
        f1, f2, f3, f4 = x
        f4_out, bound_f4 = self.REU_f4(f4, prior_0)  # b,1,12,12 b,1,48,48
        # print("f4_out, boumd_f4", f4_out.shape, bound_f4.shape)
        f4 = F.interpolate(f4_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f4 = F.interpolate(bound_f4, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)  # b,1,24,24 b,1,96,96
        # print("f3_out, boumd_f3", f3_out.shape, bound_f3.shape)
        f3 = F.interpolate(f3_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f3 = F.interpolate(bound_f3, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)  # b,1,48,48 b,1,192,192
        # print("f2_out, boumd_f2", f2_out.shape, bound_f2.shape)
        f2 = F.interpolate(f2_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f2 = F.interpolate(bound_f2, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)  # b,1,96,96 b,1,384,384
        # print("f1_out, boumd_f1", f1_out.shape, bound_f1.shape)
        f1 = F.interpolate(f1_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f1 = F.interpolate(bound_f1, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        return f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    f1 = torch.randn(2, 1, 12, 12).cuda()
    ll = torch.randn(2, 64, 96, 96).cuda()
    lh = torch.randn(2, 64, 48, 48).cuda()
    hl = torch.randn(2, 64, 24, 24).cuda()
    hh = torch.randn(2, 64, 12, 12).cuda()
    pict = torch.randn(2, 3, 384, 384).cuda()
    x = [ll, lh, hl, hh]
    rem = REM12(64, 64).cuda()
    f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = rem(x, f1, pict)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)