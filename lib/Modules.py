import torch.nn as nn
affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from lib.fcanet import MultiSpectralAttentionLayer

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.conv4 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)

        self.out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=affine_par),
            nn.PReLU(),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(32, 1, 1))


    def forward(self, input1, input2, input3, input4):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))

        if input4 is not None:
            # input4 = input4.detach()
            fuse = torch.cat([fuse, input4], dim=1)
            fuse = self.bn4(self.conv4(fuse))

        fuse = self.out(fuse)

        return fuse

    def initialize(self):
        weight_init(self)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # ReflectionPad2d(reflection_padding)是对输入的边界进行填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2  
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2] 
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, lh+hl+hh


"""
    Joint Attention module (CA + SA)
"""

class SA(nn.Module):

    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)  # C  = 1
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)  # get channel weight
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # y.squeeze(-1).transpose(-1, -2)  # N C H W -> N H W C
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)  # expand_as aim to broadcast

class AM(nn.Module):
    def __init__(self, channels, lf):
        super(AM, self).__init__()
        self.CA = MultiSpectralAttentionLayer(64, 16, 16)
        self.SA = SA(channels)

    def forward(self, x):
        x = self.CA(x)
        x = self.SA(x)
        return x

# modified by liu
class ARM(nn.Module):
    def __init__(self, channels, lf):
        super(ARM, self).__init__()
        self.CA = MultiSpectralAttentionLayer(64, 16, 16)
        self.SA = SA(channels)

    def forward(self, x):
        x1 = self.CA(x)
        x2 = self.SA(x)
        att = 1 + F.sigmoid(x1 * x2)

        return x*att

"""
    Low-Frequency Attention Module (LFA)
"""


class RB(nn.Module):
    def __init__(self, channels, lf):
        super(RB, self).__init__()
        self.RB = BasicConv2d(channels, channels, 3, padding=1, bn=nn.InstanceNorm2d if lf else nn.BatchNorm2d)

    def forward(self, x):
        y = self.RB(x)
        return y + x


class ARB(nn.Module):
    def __init__(self, channels, lf):
        super(ARB, self).__init__()
        self.lf = lf
        self.AM = AM(channels, lf)
        self.RB = RB(channels, lf)

        # self.ARM = ARM(channels, lf)  # new add

        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)

    def PONO(self, x, epsilon=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):
        if self.lf:
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))
        y = self.RB(x)
        y = self.AM(y)

        # y = self.ARM(x)  # new add
        if self.lf:
            return y * std + mean
        return y


"""
    Guidance-based Upsampling
"""

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:-r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


class AGF(nn.Module):
    def __init__(self, channels, lf):
        super(AGF, self).__init__()
        self.ARB = ARB(channels, lf)
        self.GF = GF(r=2, eps=1e-2)

    def forward(self, high_level, low_level):
        N, C, H, W = high_level.size()
        high_level_small = F.interpolate(high_level, size=(int(H / 2), int(W / 2)), mode='bilinear', align_corners=True)
        y = self.ARB(low_level)
        y = self.GF(high_level_small, low_level, high_level, y)
        return y

class AGFG(nn.Module):
    def __init__(self, channels, lf):
        super(AGFG, self).__init__()
        self.GF2 = AGF(channels, lf)
        self.GF3 = AGF(channels, lf)

    def forward(self, f2, f3, f4):

        y = self.GF2(f3, f2)
        y = self.GF3(f4, y)

        return y

class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM3, self).__init__()
        # wavelet attention module
        self.DWT = DWT()
        self.AGFG_LL = AGFG(out_channels, True)
        self.AGFG_LH = AGFG(out_channels, False)
        self.AGFG_HL = AGFG(out_channels, False)
        self.AGFG_HH = AGFG(out_channels, False)

    def forward(self, f1, f2, f3, f4):

        wf1 = self.DWT(f1)
        wf2 = self.DWT(f2)
        wf3 = self.DWT(f3)

        HH = self.AGFG_HH(wf3[3], wf2[3], wf1[3])
        return HH, f1, f2, f3, f4

"""
    LLA
"""
class LGF(nn.Module):
    def __init__(self, channels, lf):
        super(LGF, self).__init__()
        self.GF1 = AGF(channels, lf)

    def forward(self, f1, f2):
        y = self.GF1(f1, f2)

        return y

class LLA(nn.Module):
    def __init__(self, in_channels):
        super(LLA, self).__init__()
        # wavelet attention module
        self.DWT = DWT()
        self.AGFG_LL = LGF(in_channels, True)

        self.LA = ARB(in_channels, True)


        # dilation conv
        self.side_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.branch2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.branch3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv_low_f1 = BasicConv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)




    def forward(self, x, last=None):

        wf = self.DWT(x)

        if last is None:
            next = self.LA(wf[0])
        else:
            next = self.AGFG_LL(wf[0], last)

        temp = F.interpolate(next, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_low_f1(torch.cat([x, temp], dim=1))

        # x = x + temp # +
        # x = self.side_conv1(x)
        x1 = self.branch3(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)
        x = x1 * x2 + x
        # x = self.side_conv2(x)

        return x, next


"""
    Ordinary Differential Equation (ODE)
"""
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





