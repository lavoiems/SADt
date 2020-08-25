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
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=2, pad_value=1)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--data-root-tgt', type=str, help='Path to the data')
    parser.add_argument('--dataset-src', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--dataset-trg', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--max-conv-dim', type=int, default=128)
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
    N = 64
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size, max_conv_dim=args.max_conv_dim).to(device)
    generator.load_state_dict(state_dict['generator'])
    style_encoder = StyleEncoder(img_size=args.img_size).to(device)
    style_encoder.load_state_dict(state_dict['style_encoder'])

    feature_blocks = 29 if args.img_size == 256 else 8
    vgg = vgg19(pretrained=True).features[:feature_blocks].to(device)

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(data_root_src, 1, 1)[2].dataset
    dataset = getattr(images, args.dataset_trg)
    trg_dataset = dataset(data_root_tgt, 1, 1)[2].dataset

    data = []
    for i in range(N):
        idx = i
        data.append(src_dataset[idx][0])
    data = torch.stack(data).to(device)
    data = data*2 - 1

    # Infer translated images
    d_trg = torch.tensor(0==domain).repeat(N).long().to(device)
    x_idxs = torch.randint(low=0, high=len(trg_dataset), size=(N,))
    x_trg = torch.stack([trg_dataset[idx][0].to(device) for idx in x_idxs])
    x_trg = x_trg*2 - 1

    x_concat = [data]

    features = vgg(data)
    s_trg = style_encoder(x_trg, d_trg)
    x_fake = generator(data, features, s_trg)
    x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    results = [None] * len(x_concat)
    results[::2] = x_concat[:len(x_concat)//2]
    results[1::2] = x_concat[len(x_concat)//2:]
    results = torch.stack(results)
    save_image(results, 10, f'{name}.png')
