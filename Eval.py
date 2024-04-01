import argparse
import glob
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image


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

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
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
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
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
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
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
    nn.ReLU()  # relu5-4
)


class SANet(nn.Module):

    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O


import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
        self.latlayer1 = nn.Conv2d(512, 256, (1,1))
        self.latlayer2 = nn.Conv2d(512, 256, (1,1))
        self.latlayer3 = nn.Conv2d(512, 256, (1,1))

        self.channel_attention=SELayer(3 * 256)
        self.reflectPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.squeeze = nn.Conv2d(3 * 256, 512, (3, 3),padding=(0,0))


    def forward(self,feats3,feats4,feats5):
        top = feats3
        mid = feats4
        btm = feats5

        top_sample = self.latlayer1(top)
        top_sample = F.interpolate(top_sample, size=mid.size()[2:], mode='bilinear',align_corners=True)
        btm_sample = self.latlayer3(btm)
        btm_sample = F.interpolate(btm_sample, size=mid.size()[2:], mode='bilinear',align_corners=True)
        mid_sample =self.latlayer2(mid)

        result = torch.cat((top_sample,mid_sample,btm_sample),1)
        #channel wise attention
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


    def forward(self,content3_1, style3_1, content4_1, style4_1, content5_1, style5_1):
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

        ret = self.feature_pyramid(onestyle3_1 + twostyle3_1, onestyle4_1 + twostyle4_1,
                                   onestyle5_1 + twostyle5_1)

        return ret
        # return self.merge_conv(self.merge_conv_pad(
        #     self.sanet4_1(content4_1, style4_1) + self.upsample5_1(
        #         self.sanet5_1(content5_1, style5_1))))  # 把C，S的4_1，5_1经过注意力，将5_1的结果扩大两倍来匹配4_1，然后想加之后经过3*3的卷积


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='input/bird.png',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='style/feathers.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='vgg_normalised.pth')
parser.add_argument('--decoder', type=str,
                    default='./decoder_iter_41611.pth')  # ./first/decoder_iter_500000.pth    ./decoder_iter_120000.pth
parser.add_argument('--transform', type=str,
                    default='./transformer_iter_41611.pth')  # ./first/transformer_iter_500000.pth   ./transformer_iter_120000.pth

# Additional options
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args('')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = decoder
transform = Transform(in_planes=512)
vgg = vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()
data_dir = r'D:\code\SANET-master\style'  # 假设文件夹里的内容为：1.jpg  2.jpg  1.txt  2.txt
data_dirc = r'D:\code\SANET-master\input'
output = glob.glob((os.path.join(data_dir, '*.*')))
outputc = glob.glob((os.path.join(data_dirc, '*.*')))

datanames = os.listdir(data_dir)
datanamec = os.listdir(data_dirc)

name = ['1', '3314.jpg', '20200428220829.jpg', 'antimonocromatismo.jpg', 'asheville.jpg', 'brushstrokes.jpg', 'candy',
        'Composition-VII', 'contrast_of_forms.jpg', 'en_campo_gris.jpg', 'feathers', 'flower_of_life.jpg',
        'goeritz.jpg', 'impronte_d_artista.jpg', 'la_muse', 'mondrian.jpg', 'mondrian_cropped.jpg',
        'picasso_seated_nude_hr.jpg', 'picasso_self_portrait.jpg', 'rain_princess', 'scene_de_rue.jpg',
        'seated_nude', 'sketch.png', 'Starry', 'style11', 'the_resevoir_at_poitiers.jpg', 'trial.jpg', 'udnie', 'wave',
        'woman_in_peasant_dress.jpg', 'woman_in_peasant_dress_cropped.jpg', 'woman_with_hat_matisse.jpg', 'wreck.jpg']
namec = ['14.jpg', 'bair.jpg', 'bird.png', 'boat.jpg', 'chicago.jpg', 'church.jpeg', 'Content.jpg',
         'face.jpg',
         'face2.jpeg', 'flower.jpg', 'golden_gate.jpg', 'lenna.jpg', 'modern.jpg', 'newyork.jpg',
         'taj_mahal.jpg', 'tubingen.jpg', 'VaranasiCandles_lr.jpg', 'venice-boat.jpg']

with torch.no_grad():
    for j in range(2, 20):

        for i in range(6, 33):

            style = style_tf(Image.open(output[i]))
            style = style.to(device).unsqueeze(0)
            content = content_tf(Image.open(outputc[j]))
            content = content.to(device).unsqueeze(0)
            x = 0
            for x in range(args.steps):
                print('iteration ' + str(x))

                Content3_1 = enc_3(enc_2(enc_1(content)))
                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                Content5_1 = enc_5(Content4_1)

                Style3_1 = enc_3(enc_2(enc_1(style)))
                Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                Style5_1 = enc_5(Style4_1)

                content = decoder(transform(Content3_1, Style3_1,Content4_1, Style4_1, Content5_1, Style5_1))

                content.clamp(0, 255)

            content = content.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output,datanamec[j],
                datanames[i], args.save_ext
            )
            save_image(content, output_name)

# content = content_tf(Image.open(args.content))
# style = style_tf(Image.open(args.style))
#
# style = style.to(device).unsqueeze(0)
# content = content.to(device).unsqueeze(0)
#
# with torch.no_grad():
#
#     for i in range(0,12):
#
#         for x in range(args.steps):
#
#             print('iteration ' + str(x))
#
#             Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
#             Content5_1 = enc_5(Content4_1)
#
#             Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
#             Style5_1 = enc_5(Style4_1)
#
#             content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
#
#             content.clamp(0, 255)
#
#         content = content.cpu()
#
#         output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
#                     args.output, splitext(basename(args.content))[0],
#                     splitext(basename(args.style))[0], args.save_ext
#                 )
#         save_image(content, output_name)
