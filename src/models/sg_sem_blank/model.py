"""
Code adapted from the StarGAN v2: https://github.com/clovaai/stargan-v2
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from common.initialize import define_last_model


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, bottleneck_blocks=2, bottleneck_size=64):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - int(np.log2(bottleneck_size/4))
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(bottleneck_blocks):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, d):
        o = z
        h = self.shared(o)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(d.size(0))).to(d.device)
        s = out[idx, d]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, n_unshared_layers=0):
        super().__init__()
        self.num_domains = num_domains
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            unshared = []
            for _ in range(n_unshared_layers):
                unshared += [nn.Linear(dim_out, dim_out),
                                  nn.LeakyReLU(0.2)]
            unshared += [nn.Linear(dim_out, style_dim)]
            self.unshared += [nn.Sequential(*unshared)]

    def forward(self, x, d):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        pos = d
        idx = torch.LongTensor(range(d.size(0))).to(x.device)
        s = out[idx, pos]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, nc=5):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.main = nn.Sequential(*blocks)
        self.dis = nn.Conv2d(dim_out, num_domains, 1, 1, 0)
        self.classifier = nn.ModuleList([nn.Linear(dim_out, nc) for _ in range(num_domains)])

    def forward(self, x, d):
        out = self.main(x)
        out = self.dis(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(d.size(0))).to(d.device)
        out = out[idx, d]  # (batch)
        return out

    def classify(self, x, d):
        h = self.main(x)
        h = h.squeeze()
        out = []
        for layer in self.classifier:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(d.shape[0])).to(d.device)
        out = out[idx, d]
        return out


def build_model(args):
    generator = Generator(args.img_size, args.style_dim, args.max_conv_dim, args.bottleneck_blocks, args.bottleneck_size)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains, nc=args.num_classes)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)
        return x


class Classifier(nn.Module):
    def __init__(self, h_dim, nc, z_dim, **kwargs):
        super(Classifier, self).__init__()
        self.x = nn.Sequential(nn.Linear(z_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Dropout(0.5),
                               GaussianLayer(),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True),
                               nn.Dropout(0.5),
                               GaussianLayer())

        self.mlp = nn.Sequential(nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(h_dim, nc),
                                 nn.LayerNorm(nc, elementwise_affine=False, eps=1e-3),
                                 )

    def forward(self, x):
        o = self.x(x)
        return self.mlp(o)


def ss_model(ss_path):
    ss = torchvision.models.resnet50()
    ss.fc = torch.nn.Identity()
    state_dict = torch.load(ss_path, map_location='cpu')['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    err = ss.load_state_dict(state_dict, strict=False)
    #print(err)
    ss.eval()
    return ss


class Semantics(nn.Module):
    def __init__(self, ss, cluster):
        super().__init__()
        self.ss = ss
        self.cluster = cluster

    def forward(self, x):
        o = self.ss(x)
        o = self.cluster(o)
        return o


def semantics(ss_path, cluster_type, cluster_path, **kwargs):
    if ss_path:
        ss = ss_model(ss_path)
        cluster = define_last_model(cluster_type, cluster_path, 'classifier', **kwargs)
        return Semantics(ss, cluster)
    else:
        return define_last_model(cluster_type, cluster_path, 'classifier', **kwargs)
