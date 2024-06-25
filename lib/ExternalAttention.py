import torch
from torch import nn
from torch.nn import init
import math
import torch.nn.functional as F



# class ExternalAttention(nn.Module):
#
#     def __init__(self, d_model, S=64):
#         super().__init__()
#         self.mk = nn.Linear(d_model, S, bias=False)
#         self.mv = nn.Linear(S, d_model, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, queries):
#         attn = self.mk(queries)  # bs,n,S
#         attn = self.softmax(attn)  # bs,n,S
#         attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
#         out = self.mv(attn)  # bs,n,d_model
#
#         return out


# 输入 B C N,  输出 B C N

class ExternalAttention(nn.Module):

    def __init__(self, d_model=64, S=64):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n)  # 即bs，n，d_model
        queries = queries.permute(0, 2, 1)
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w)
        x = x + x_attn
        x = F.relu(x)
        return x




if __name__ == '__main__':
    block = ExternalAttention(d_model=64, S=8).cuda()
    input = torch.rand(64, 144, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
