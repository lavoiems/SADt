import os
import torch
from ..model import Generator, MappingNetwork, ss_model, cluster_model
import torchvision.utils as vutils
from common.loaders import images


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--dataset-src', type=str, help='Path to the data')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model-path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')


@torch.no_grad()
def execute(args):
    state_dict_path = args.state_dict_path
    domain = args.domain
    ss_path = args.ss_path
    da_path = args.da_path
    name = args.save_name

    device = 'cuda'
    N = 64
    latent_dim = 16
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork(nc=10)
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    ss = ss_model(ss_path).cuda()
    da = cluster_model(da_path).cuda()

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(args.data_root_src, 1, 1)[2].dataset

    data = []
    for i in range(N):
        idx = i
        data.append(src_dataset[idx][0])
    data = torch.stack(data).to(device)
    data = data*2 - 1

    y_src = da(ss((data+1)*0.5)).argmax(1)
    print(y_src)

    # Infer translated images
    d_trg = torch.tensor(0==domain).repeat(N).long().to(device)
    z_trg = torch.randn(N, latent_dim).to(device)
    print(z_trg.shape, data.shape, y_src.shape)

    x_concat = [data]

    print(z_trg.shape, y_src.shape, d_trg.shape)
    s_trg = mapping(z_trg, y_src, d_trg)
    print(data.shape, s_trg.shape)
    x_fake = generator(data, s_trg)
    x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    results = [None] * len(x_concat)
    results[::2] = x_concat[:len(x_concat)//2]
    results[1::2] = x_concat[len(x_concat)//2:]
    results = torch.stack(results)
    save_image(results, 10, f'{name}.png')
