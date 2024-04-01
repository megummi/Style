import torch


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


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)  # 初始化sanet类，输入通道512，3_1通道的输入
        self.sanet5_1 = SANet(in_planes=in_planes)  # 4_1通道的输入
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样扩大两倍，因为3_1，4_1的宽和高不匹配
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))  # padding填充
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))  # 3*3卷积

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) + self.upsample5_1(
                self.sanet5_1(content5_1, style5_1))))  # 把C，S的4_1，5_1经过注意力，将5_1的结果扩大两倍来匹配4_1，然后想加之后经过3*3的卷积


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
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False  # 将enc_1, enc_2, enc_3, enc_4, enc_5这几个参数不更新留着备用

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

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)  # 提取风格图像的enc_1, enc_2, enc_3, enc_4, enc_5的值
        content_feats = self.encode_with_intermediate(content)  # 提取内容图像的enc_1, enc_2, enc_3, enc_4, enc_5的值
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4],
                                  style_feats[4])  # 返回C，S的4_1，5_1经过风格转换加注意力的图
        g_t = self.decoder(stylized)  # 将结果解码
        g_t_feats = self.encode_with_intermediate(g_t)  # 解码后再次提取enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5的值，即再次经过vgg解码
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)  # 计算内容损失
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])  # 计算风格损失
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])  # 五个层次的损失全部计算
        """IDENTITY LOSSES"""  # 计算id损失，即全局损失
        Icc = self.decoder(
            self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))  # 内容图像经过vgg及注意力
        Iss = self.decoder(
            self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))  # 风格图像经过vgg和注意力
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)  # 再次经过解码
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2


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
parser.add_argument('--batch_size', type=int, default=5)  # 批次
parser.add_argument('--style_weight', type=float, default=3.0)  # 风格权重
parser.add_argument('--content_weight', type=float, default=1.0)  # 内容权重
parser.add_argument('--n_threads', type=int, default=16)  # 线程数量
parser.add_argument('--save_model_interval', type=int, default=1000)  # 每1000迭代存储一次
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

if (args.start_iter > 0):  # 当不是第0次迭代的时候，加载参数
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):  # tqdm实现进度条功能
    adjust_learning_rate(optimizer, iteration_count=i)  # 调增学习率
    content_images = next(content_iter).to(device)  # next函数抽取可迭代的对象
    style_images = next(style_iter).to(device)
    loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)  # 经过网络返回损失
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1  # 总损失

    optimizer.zero_grad()  # 梯度回传
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:  # 当迭代次数达到存储次数或者上限次数时存储参数
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
