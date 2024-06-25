from lib.Modules import GCM3, BasicConv2d, LLA

import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.Newdecoder import REM12
# from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.swin_encoder import SwinTransformer

from lib.Rfb import RFB_modified
from lib.Modules import Conv_Block

class Network(nn.Module):
    def __init__(self, channels=64, pretrained=True):
        super(Network, self).__init__()

        self.encoder = SwinTransformer(img_size=384,
                                       embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=12)
        if pretrained:
            pretrained_dict = torch.load('/home/ljh/ljh_COD/Net/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)

            # frozen the encoder
            # for p in self.encoder.parameters():
            #     p.requires_grad = False

        self.GCM3 = GCM3(256, channels)

        self.LLA_4 = LLA(channels)
        self.LLA_3 = LLA(channels)
        self.LLA_2 = LLA(channels)
        self.LLA_1 = LLA(channels)

        self.REM11 = REM12(channels, channels)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

        self.conv_low_f1 = BasicConv2d(channels * 2, 64, kernel_size=3, stride=1, padding=1)

        # rfb
        self.rfb1 = RFB_modified(channels * 2, channels)
        self.rfb2 = RFB_modified(channels * 4, channels)
        self.rfb3 = RFB_modified(channels * 8, channels)
        self.rfb4 = RFB_modified(channels * 16, channels)

        self.convblock = Conv_Block(channels)


    def forward(self, x):

        image = x
        features = self.encoder(x)

        x4 = features[1]
        x3 = features[2]
        x2 = features[3]
        x1 = features[4]


        # rfb modified
        f1 = self.rfb1(x1)
        f2 = self.rfb2(x2)
        f3 = self.rfb3(x3)
        f4 = self.rfb4(x4)

        # -----------------high frequency-----------------
        HH, f1, f2, f3, f4 = self.GCM3(f1, f2, f3, f4)

        # -----------------low frequency-----------------
        _, f4_ll = self.LLA_4(f4)
        _, f3_ll = self.LLA_3(f3, last=f4_ll)
        _, f2_ll = self.LLA_2(f2, last=f3_ll)
        _, f1_ll_g = self.LLA_1(f1, last=f2_ll)

        f1_ll = F.interpolate(f1_ll_g, scale_factor=2, mode='bilinear', align_corners=False)
        f1_l = self.conv_low_f1(torch.cat([f1_ll, f1], dim=1))

        prior_cam = self.convblock(F.interpolate(f4, size=f3.size()[2:], mode='bilinear'), f3, F.interpolate(f2, size=f3.size()[2:], mode='bilinear'), f2_ll)

        # f1 combine
        HH_up = self.dePixelShuffle(HH)
        f1_HH = torch.cat([HH_up, f1_l], dim=1)      # f1-> f1_l
        f1_HH = self.one_conv_f1_hh(f1_HH)

        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)

        # AEED
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1_HH, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(64).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)