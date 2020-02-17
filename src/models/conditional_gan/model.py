import math
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, nc, **kwargs):
        super(Critic, self).__init__()
        assert kernel_dim % 16 == 0, "kernel_dim has to be a multiple of 16"

        x = [nn.Conv2d(i_dim, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True)]

        dim = h_dim
        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        for _ in range(n_layers):
            in_dim = dim
            dim *= 2
            x += [nn.Conv2d(in_dim, dim, 4, 2, 1),
                  nn.LeakyReLU(0.2, inplace=True)]
        x += [nn.Conv2d(dim, dim, 4, 1, 0)]
        self.x = nn.Sequential(*x)
        z = [nn.Linear(nc, dim),
             nn.LeakyReLU(inplace=True),
             nn.Linear(dim, dim),
             nn.LeakyReLU(inplace=True),
             nn.Linear(dim, dim),
             nn.LeakyReLU(inplace=True)]
        self.z = nn.Sequential(*z)
        self.out = nn.Sequential(nn.Linear(dim*2, 512),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512, 512),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(h_dim, 1))

    def forward(self, x, z):
        ox = self.x(x).squeeze()
        oz = self.z(z.squeeze())
        o = torch.cat((ox, oz), 1)
        return self.o(o)


class Generator(nn.Module):
    def __init__(self, o_dim, kernel_dim, z_dim, h_dim, nc, **kwargs):
        super(Generator, self).__init__()

        n_layers = int(math.log(kernel_dim//4, 2)) - 1
        dim = h_dim * n_layers**2
        decoder = [nn.Conv2d(z_dim+nc, dim, 1, 1, 0),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(dim, dim, 1, 1, 0),
                   nn.ReLU(inplace=True),
                   nn.ConvTranspose2d(dim, dim, 4, 1, 0),
                   nn.BatchNorm2d(dim),
                   nn.ReLU(True)]

        for _ in range(n_layers):
            in_dim = dim
            dim //= 2
            decoder += [nn.ConvTranspose2d(in_dim, dim, 4, 2, 1),
                        nn.BatchNorm2d(dim),
                        nn.ReLU(True)]

        decoder += [nn.ConvTranspose2d(dim, o_dim, 4, 2, 1),
                    nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z, z2):
        x = torch.cat((z, z2), 1).view(z.shape[0], -1, 1, 1)
        return self.decoder(x)
