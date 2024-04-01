import torch
from torch.nn import functional as F


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


import torch.nn as nn

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(  # 输入1*3*512*512
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),  # 1 64 512
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 1 64 256
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),  # 1 128 256
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 1 128 128
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),  # 1 256 128
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 1 256 64
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),  # 1 512 64
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 1 512 32
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4 # 1 512 32*32
)


class SANet(nn.Module):  # sa注意模块
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.sm = nn.Softmax(dim=-1)  # softmax操作
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))  # 输出之后经过1*1的卷积

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))  # 上面已经初始化完毕，现在相当于调用conv2d的forward函数，先norm化在经过1*1的卷积
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)  # 改变形状，并且行列转置 Fc‘
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)  # 改变g的形状，Fs'
        S = torch.bmm(F, G)  # Fs‘和Fc’相乘
        S = self.sm(S)  # softmax操作
        b, c, h, w = H.size()  # 最下面的Fs
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))  # Fs‘和Fc’相乘的结果在与Fs相乘
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)  # 重塑为输入的形状
        O = self.out_conv(O)  # 输出经过1*1的卷积
        O += content  # 相当于跳跃链接
        return O


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ContextualLoss import ContextualLoss


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Aattn(nn.Module):
    def __init__(self, in_planes):
        super(Aattn, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))  # 先经过1*1的卷积
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(content)
        G = self.g(style)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()

        style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2)
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        S = self.sm(S)
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        O = std * mean_variance_norm(content) + mean
        O = O.view(b, _, h, w)
        O = self.out_conv(O)
        O += content
        return O


class MAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 这里的dim指的是c，即input输入N个数据的维度。
        # 另外，为避免繁琐，将8个头的权重拼在一起，而三个不同的权重由后面的Linear生成。
        # 而self.qkv的作用是是将input X (N,C)与权重W（C，8*Ｃ1*3）相乘得到Q_K_V拼接在一起的矩阵。
        # 所以,dim*3表示的就是所有头所有权重拼接的维度，即8*Ｃ1*3。即dim=C=C1*3。
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # bias默认为True，为了使该层学习额外的偏置。
        self.attn_drop = nn.Dropout(attn_drop)
        # dropout忽略一半节点
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B为batch_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将x构造成一个（B，N，3，8，c1）的一个矩阵，然后将其变成（3，B，8，N，c1）。
        # 是为了使得后面能将其分为三部分，分别作为不同的权重，维度为（B，8，N，c1）
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 将k的维度从（B，8，N，c1）转变为（B，8，c1，N），其实就是对单头key矩阵转置，使Query*key^T，得到scores结果，然后×self.scale，即×C1的-0.5。
        # 就是做一个归一化。乘一个参数，不一定是这个值。
        # 维度为（B，8，N，N）
        attn = attn.softmax(dim=-1)
        # 对归一化的scores做softmax
        # 维度为（B，8，N，N）
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 将scores与values矩阵相乘，得到数据维度为（B，8，N，C1），使用transpose将其维度转换为（B，N，8，C1）
        x = self.proj(x)
        # 做一个全连接
        x = self.proj_drop(x)
        # 做Dropout
        return x
        # 得到多头注意力所有头拼接的结果。


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Feature_pyramid(nn.Module):
    def __init__(self):
        super(Feature_pyramid, self).__init__()
        self.latlayer1 = nn.Conv2d(512, 256, (1, 1))
        self.latlayer2 = nn.Conv2d(512, 256, (1, 1))
        self.latlayer3 = nn.Conv2d(512, 256, (1, 1))

        self.channel_attention = SELayer(3 * 256)
        self.reflectPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.squeeze = nn.Conv2d(3 * 256, 512, (3, 3), padding=(0, 0))

    def forward(self, feats3, feats4, feats5):
        top = feats3
        mid = feats4
        btm = feats5

        top_sample = self.latlayer1(top)
        top_sample = F.interpolate(top_sample, size=mid.size()[2:], mode='bilinear', align_corners=True)
        btm_sample = self.latlayer3(btm)
        btm_sample = F.interpolate(btm_sample, size=mid.size()[2:], mode='bilinear', align_corners=True)
        mid_sample = self.latlayer2(mid)

        result = torch.cat((top_sample, mid_sample, btm_sample), 1)
        # channel wise attention
        result = self.channel_attention(result)
        result = self.reflectPad(result)
        result = self.squeeze(result)
        return result


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = Aattn(in_planes=in_planes)  # 初始化sanet类，输入通道512，3_1通道的输入
        self.sanet5_1 = Aattn(in_planes=in_planes)  # 4_1通道的输入
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样扩大两倍，因为3_1，4_1的宽和高不匹配
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))  # padding填充
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))  # 3*3卷积
        self.canet = CoordAtt(in_planes, in_planes)
        self.ecanet = eca_block(in_planes)
        self.feature_pyramid = Feature_pyramid()
        self.sanet3_1 = Aattn(in_planes=in_planes)
        self.conv = nn.Conv2d(256, in_planes, 1)
        self.originsanet3_1 = SANet(in_planes=in_planes)
        self.originsanet4_1 = SANet(in_planes=in_planes)
        self.originsanet5_1 = SANet(in_planes=in_planes)

    def forward(self, content3_1, style3_1, content4_1, style4_1, content5_1, style5_1):
        content3_1 = self.conv(content3_1)
        style3_1 = self.conv(style3_1)

        content3_1 = self.canet(content3_1)
        content4_1 = self.canet(content4_1)
        content5_1 = self.canet(content5_1)

        style3_1 = self.ecanet(style3_1)
        style4_1 = self.ecanet(style4_1)
        style5_1 = self.ecanet(style5_1)

        onestyle3_1 = self.originsanet3_1(content3_1, style3_1)
        onestyle4_1 = self.originsanet4_1(content4_1, style4_1)
        onestyle5_1 = self.originsanet5_1(content5_1, style5_1)

        twostyle3_1 = self.sanet3_1(content3_1, style3_1)
        twostyle4_1 = self.sanet4_1(content4_1, style4_1)
        twostyle5_1 = self.sanet5_1(content5_1, style5_1)

        ret = self.feature_pyramid(onestyle3_1+twostyle3_1, onestyle4_1+twostyle4_1,
                                   onestyle5_1+twostyle5_1)

        return ret
        # return self.merge_conv(self.merge_conv_pad(
        #     self.sanet4_1(content4_1, style4_1) + self.upsample5_1(
        #         self.sanet5_1(content5_1, style5_1))))  # 把C，S的4_1，5_1经过注意力，将5_1的结果扩大两倍来匹配4_1，然后想加之后经过3*3的卷积


class Net(nn.Module):  # 本文的神经网络
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())  # 将编码器的网络层列出来，vgg19
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1   # 前4层的操作，即输出relu1_1，[0：4]左闭右开0，1，2，3
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1  第4，5，6，7，8，9，10层 得到2_1的输出
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1  第11.12.13.14.15.16.17层 得到3_1的输出
        self.enc_4 = nn.Sequential(
            *enc_layers[18:31])  # relu3_1 -> relu4_1  第18.19.20.21.22.23.24.25.26.27.28.29.30层 得到4_1的输出
        self.enc_5 = nn.Sequential(
            *enc_layers[31:44])  # relu4_1 -> relu5_1  第31.32.33.34.35.36.37.38.39.40.41.42.43层 得到5_1的输出
        # transform
        self.transform = Transform(in_planes=512)  # 初始化transform类，输入通道为512
        self.decoder = decoder  # decoder
        if (start_iter > 0):  # 如果迭代次数不为0，即之前训练过
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))  # 将之前参数同步
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))  # 将之前参数同步
        self.mse_loss = nn.MSELoss()  # 均方误差
        self.sm = nn.Softmax(dim=-1)
        self.cx = ContextualLoss()
        self.f = nn.Conv2d(512, 512, (1, 1))  # 先经过1*1的卷积
        self.g = nn.Conv2d(512, 512, (1, 1))  # 先经过1*1的卷积
        self.h = nn.Conv2d(512, 512, (1, 1))
        self.conv = nn.Conv2d(256, 512, 1)
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False  # 将enc_1, enc_2, enc_3, enc_4, enc_5这几个参数不更新留着备用
        # lap
        self.avg_pool = nn.AvgPool2d(3, stride=3)
        self.laplacian_filter = nn.Conv2d(3, 3, 3, 1, bias=False)
        laplacian = torch.tensor([[[0., -1., 0.],
                                   [-1., 4., -1.],
                                   [0., -1., 0.]],
                                  [[0., -1., 0.],
                                   [-1., 4., -1.],
                                   [0., -1., 0.]],
                                  [[0., -1., 0.],
                                   [-1., 4., -1.],
                                   [0., -1., 0.]],
                                  ])
        laplacian = laplacian.unsqueeze(0)
        self.laplacian_filter.weight = nn.Parameter(laplacian)
        self.laplacian_filter.weight.requires_grad = False
        self.originsanet3_1 = SANet(512)
        self.originsanet4_1 = SANet(512)
        self.originsanet5_1 = SANet(512)

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):  # 0,1,2,3,4
            func = getattr(self, 'enc_{:d}'.format(
                i + 1))  # getattr(x,y) 函数用于返回一个对象属性值,返回x对象的y属性值,即返回self的enc_1, enc_2, enc_3, enc_4, enc_5的值
            results.append(func(results[-1]))
        return results[1:]  # 提取enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5的值

    def calc_content_loss(self, input, target, norm=False):  # 计算内容损失
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):  # 计算风格损失
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def contextual_loss(self, cs_feats, style_feats):
        loss_cx = 0.0
        for i in range(3, 5):
            loss_cx += self.cx(cs_feats[i], style_feats[i])
        return loss_cx

    def lap(self, x):
        x = self.avg_pool(x)
        x = self.laplacian_filter(x)
        return x

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)  # 提取风格图像的enc_1, enc_2, enc_3, enc_4, enc_5的值
        content_feats = self.encode_with_intermediate(content)  # 提取内容图像的enc_1, enc_2, enc_3, enc_4, enc_5的值
        stylized = self.transform(content_feats[2], style_feats[2], content_feats[3], style_feats[3], content_feats[4],
                                  style_feats[4])  # 返回C，S的4_1，5_1经过风格转换加注意力的图
        g_t = self.decoder(stylized)  # 将结果解码
        g_t_feats = self.encode_with_intermediate(g_t)  # 解码后再次提取enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5的值，即再次经过vgg编码
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)  # 计算内容损失
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])  # 计算风格损失
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])  # 五个层次的损失全部计算

        #
        contlap = self.lap(content)
        stylap = self.lap(g_t)
        loss_lap = self.mse_loss(contlap, stylap)
        #

        for i in range(2, 5):
            if i == 2:
                content3_1 = self.conv(content_feats[i])
                content3 = self.conv(content_feats[i])
                style3_1 = self.conv(style_feats[i])

                sacontent3_1 = self.conv(content_feats[i])
                sastyle3_1 = self.conv(style_feats[i])

                mytarget = g_t_feats[i]
                mytargets = self.conv(mytarget)
                F = self.f(content3_1)
                G = self.g(style3_1)
                H = self.h(style3_1)
                b, _, h_g, w_g = G.size()
                G = G.view(b, -1, w_g * h_g).contiguous()

                style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2)
                b, _, h, w = F.size()
                F = F.view(b, -1, w * h).permute(0, 2, 1)
                S = torch.bmm(F, G)
                S = self.sm(S)
                mean = torch.bmm(S, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

                sastyled3_1 = self.originsanet3_1(sacontent3_1, sastyle3_1)

                saLf2 = (self.mse_loss(mytargets, sastyled3_1))/20
                Lf2 = self.mse_loss(mytargets, std * mean_variance_norm(content3) + mean) + saLf2

                continue

            mytarget = g_t_feats[i]
            F = self.f(content_feats[i])
            G = self.g(style_feats[i])
            H = self.h(style_feats[i])
            b, _, h_g, w_g = G.size()
            G = G.view(b, -1, w_g * h_g).contiguous()

            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2)
            b, _, h, w = F.size()
            F = F.view(b, -1, w * h).permute(0, 2, 1)
            S = torch.bmm(F, G)
            S = self.sm(S)
            mean = torch.bmm(S, style_flat)
            # std: b, n_c, c
            std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
            # mean, std: b, c, h, w
            mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

            sastyled = self.originsanet3_1(content_feats[i], style_feats[i])

            if i == 3:
                Lf3 = self.mse_loss(mytarget, std * mean_variance_norm(content_feats[i]) + mean) + (self.mse_loss(mytarget, sastyled))/20
            if i == 4:
                Lf4 = self.mse_loss(mytarget, std * mean_variance_norm(content_feats[i]) + mean) + Lf3 + Lf2 + (self.mse_loss(mytarget, sastyled))/20

        loss_cx = self.contextual_loss(g_t_feats, style_feats)

        """IDENTITY LOSSES"""  # 计算id损失，即全局损失
        Icc = self.decoder(
            self.transform(content_feats[2], style_feats[2], content_feats[3], content_feats[3], content_feats[4],
                           content_feats[4]))  # 内容图像经过vgg及注意力
        Iss = self.decoder(
            self.transform(content_feats[2], style_feats[2], style_feats[3], style_feats[3], style_feats[4],
                           style_feats[4]))  # 风格图像经过vgg和注意力
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)  # 再次经过解码
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2, Lf4, loss_cx, loss_lap


import numpy as np
from torch.utils import data


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter, writer
from torchvision import transforms
from tqdm import tqdm

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(
    data.Dataset):  # data.Dataset一个数据集抽象类，它是其他所有数据集类的父类（所有其他数据集类都应该继承它），继承时需要重写方法 __len__ 和 __getitem__
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):  # __getitem__ 是可以通过索引号找到数据的方法
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):  # __len__ 是提供数据集大小的方法，
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./train2014',  # 内容图像目录
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='./train',  # 风格图像目录
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')  # vgg的模型

# training options
parser.add_argument('--save_dir', default='./experiments',  # 存储目录
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',  # 日志目录
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)  # 学习率
parser.add_argument('--lr_decay', type=float, default=5e-5)  # 学习率的衰减
parser.add_argument('--max_iter', type=int, default=160000)  # 最大步数
parser.add_argument('--batch_size', type=int, default=1)  # 批次
parser.add_argument('--style_weight', type=float, default=3.0)  # 风格权重
parser.add_argument('--content_weight', type=float, default=1.0)  # 内容权重
parser.add_argument('--n_threads', type=int, default=1)  # 线程数量
parser.add_argument('--save_model_interval', type=int, default=2500)  # 每1000迭代存储一次
parser.add_argument('--start_iter', type=float, default=0)  # 开始迭代的次数
args = parser.parse_args('')

device = torch.device('cuda')

decoder = decoder  # 解码器
vgg = vgg  # 预训练的vgg

vgg.load_state_dict(torch.load(args.vgg))  # 加载预训练的vgg的参数
vgg = nn.Sequential(*list(vgg.children())[:44])  # 采用vgg的前0-44层即特征编码不涉及线性分类
network = Net(vgg, decoder, args.start_iter)  # 初始化Net网络，encoder是vgg，decoder是自设的decoder
network.train()  # 设置为训练模式
network.to(device)  # 按cuda训练

content_tf = train_transform()  # 重塑为512*512，并按照256随机裁剪，然后转化为张量形式
style_tf = train_transform()  #

content_dataset = FlatFolderDataset(args.content_dir, content_tf)  # 读取数据集的图片，并将其向量化
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(  # iter（）生成迭代器，DataLoader数据集加载器
    content_dataset, batch_size=args.batch_size,  # 采样，默认设置为None。根据定义的策略从数据集中采样输入
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([  # 优化器要优化的参数
    {'params': network.decoder.parameters()},
    {'params': network.transform.parameters()}], lr=args.lr)
max = 80

if (args.start_iter > 0):  # 当不是第0次迭代的时候，加载参数
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):  # tqdm实现进度条功能
    adjust_learning_rate(optimizer, iteration_count=i)  # 调增学习率
    content_images = next(content_iter).to(device)  # next函数抽取可迭代的对象
    style_images = next(style_iter).to(device)
    loss_c, loss_s, l_identity1, l_identity2, l_f, l_cx, l_lap = network(content_images, style_images)  # 经过网络返回损失
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1 + 3 * l_f + l_cx * 3 + 5 * l_lap  # 总损失

    optimizer.zero_grad()  # 梯度回传
    loss.backward()
    optimizer.step()

    if i + 1 > 10:

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter or float(
                loss.item()) < max:  # 当迭代次数达到存储次数或者上限次数时存储参数

            if (float(loss.item()) < max):
                max = float(loss.item())
            l_id = l_identity1 * 50 + l_identity2 * 1
            print('  ')
            print('第%d次' % (i + 1))
            print('总损失:%.3f' % float(loss.item()))

            fp = open("./loss.text", "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
            # print('  ', file=fp)
            # print('第%d次' % (i + 1), file=fp)
            # print('总损失:%.3f' % float(loss.item()), file=fp)
            # print('内容损失:%.3f 风格损失:%.3f 全局损失:%.3f 网络损失:%.3f 上下文余弦:%.3f 拉普拉斯损失:%.3f'
            #       % (float(loss_c.item()), float(loss_s.item()), float(l_id.item()), float((3 * l_f).item()),
            #          float((l_cx * 3).item()), float(500 * l_lap.item())), file=fp)
            # print('  ', file=fp)

            print((i + 1), float(loss.item()), float(loss_c.item()), float(loss_s.item()), float(l_id.item()),
                  float((3 * l_f).item()),
                  float((l_cx * 3).item()), float(500 * l_lap.item()), file=fp)
            fp.close()

            state_dict = decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                       '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
            state_dict = network.transform.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                       '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                               i + 1))
            state_dict = optimizer.state_dict()
            torch.save(state_dict,
                       '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                             i + 1))
writer.close()
