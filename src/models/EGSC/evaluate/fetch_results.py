import os
import torch
from ..model import Generator, StyleEncoder
import torchvision.utils as vutils
from common.loaders import images
from torchvision.models import vgg19


def save_image(x, ncol, filename):
    print(x.min(), x.max())
    x.clamp_(-1, 1)
    x = (x + 1) / 2
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--data-root-tgt', type=str, help='Path to the data')
    parser.add_argument('--dataset-src', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--dataset-trg', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')
    parser.add_argument('--img-size', type=int, default=256, help='Size of the image')
    parser.add_argument('--bottleneeck-size', type=int, default=64, help='Size of the bottleneck')
    parser.add_argument('--bottleneck_blocks', type=int, default=4, help='Number of layers at the bottleneck')


@torch.no_grad()
def execute(args):
    state_dict_path = args.state_dict_path
    data_root_src = args.data_root_src
    data_root_tgt = args.data_root_tgt
    domain = args.domain
    name = args.save_name

    device = 'cuda'
    N = 5
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size).to(device)
    generator.load_state_dict(state_dict['generator'])
    style_encoder = StyleEncoder(img_size=args.img_size).to(device)
    style_encoder.load_state_dict(state_dict['style_encoder'])

    feature_blocks = 29 if args.img_size == 256 else 8
    vgg = vgg19(pretrained=True).features[:feature_blocks].to(device)

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(data_root_src)
    dataset = getattr(images, args.dataset_trg)
    trg_dataset = dataset(data_root_tgt)

    # This is for sketch-real
    idxs = [0, 15, 31, 50, 60]
    data = []
    for i in range(N):
        idx = idxs[i]
        data.append(src_dataset[idx])
    data = torch.stack(data).to(device)

    # Infer translated images
    d_trg = torch.tensor(0==domain).repeat(25).long().to(device)
    x_idxs = torch.randint(low=0, high=len(trg_dataset), size=(5,))
    x_trg = [trg_dataset[idx].to(device) for idx in x_idxs]
    x_trg = torch.stack(5*x_trg)
    tmp_x = dataset[0]
    x_trg = [x_trg[0::5], x_trg[1::5], x_trg[2::5], x_trg[3::5], x_trg[4::5]]
    x_trg = torch.cat(x_trg)
    save_image(x_trg, 5, f'style_imgs:25.png')
    data = torch.cat(5*[data])
    save_image(data, 5, f'data:25.png')

    N, C, H, W = data.size()
    x_concat = [data]

    features = vgg(data)
    s_trg = style_encoder(x_trg, d_trg)
    x_fake = generator(data, features, s_trg)
    x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    print(x_concat[:5].shape, x_concat[N:].shape)
    results = torch.cat([x_concat[:5], x_concat[N:]])
    save_image(results, 5, f'{name}.png')
