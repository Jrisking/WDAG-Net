import torch.nn as nn
from lib.Modules import BasicConv2d
from lib.CBAM import SpatialAttention

class MFA(nn.Module):
    def __init__(self):
        super(MFA, self).__init__()
        self.fx = BasicConv2d(64, 64, 3, stride=1, padding=1, need_relu=False)
        self.fx1 = BasicConv2d(64, 64, 3, stride=1, padding=1, need_relu=False)
        self.fx2 = BasicConv2d(64, 64, 3, stride=1, padding=1, need_relu=False)

        self.f1_r2 = BasicConv2d(64, 64, 3, 1, 2, dilation=2, need_relu=False) # need_relu=False
        # self.f2_r4 = BasicConv2d(64, 64, 3, 1, 3, dilation=3, need_relu=False)

        self.out = BasicConv2d(64, 64, 3, stride=1, padding=1)
        # self.out1 = BasicConv2d(64, 64, 3, stride=1, padding=1, need_relu=False)
        self.SA = SpatialAttention(3)



    def forward(self, x, last):
        # x = self.fx(x)
        last = self.SA(last)

        f1 = self.fx1(x)
        f1 = self.f1_r2(f1) * last

        # f2 = self.fx2(x)
        # f2 = self.f1_r2(f2) * last

        out = self.out(f1) + x
        # out = self.f1_r2(x) * last + x


        return out
