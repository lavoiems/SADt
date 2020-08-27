import os
import torch
from ..model import Generator, MappingNetwork, semantics
from torch.utils import data
from common.util import save_image, normalize
from common.loaders import images
from common.evaluation import fid


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path of the model')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--data-root-src', type=str, help='Path of the data')
    parser.add_argument('--data-root-tgt', type=str, help='Path of the data')
    parser.add_argument('--dataset-src', type=str, help='Dataset in {dataset_single, dataset_mnist, dataset_svhn}')
    parser.add_argument('--dataset-tgt', type=str, help='Dataset in {dataset_single, dataset_mnist, dataset_svhn}')


@torch.no_grad()
def execute(args):
    device = 'cuda'
    latent_dim = 16
    batch_size = 128
    # Load model
    state_dict = torch.load(args.state_dict_path, map_location='cpu')

    generator = Generator(bottleneck_size=64, bottleneck_blocks=4).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork()
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    sem = semantics(args.ss_path, args.model_type, args.da_path, nc=args.nc).to(device)

    dataset = getattr(images, args.dataset_src)(args.dataroot_src)
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = getattr(images, args.dataset_tgt)(args.dataroot_tgt)
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)

    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    print('Fetching generated data')
    d = torch.tensor(args.domain).repeat(batch_size).long().to(device)
    for data in src:
        data = data.to(device)
        data = (data+1)/2
        d_trg = d[:data.shape[0]]
        y_trg = sem(data).argmax(1)
        for i in range(10):
            z_trg = torch.randn(data.shape[0], latent_dim, device=device)
            s_trg = mapping(z_trg, y_trg, d_trg)
            gen = generator(data, s_trg)
            generated.append(gen)
    generated = torch.cat(generated)
    generated = normalize(generated)
    save_image(generated[:4], 'Debug.png')

    print('Fetching target data')
    trg_data = []
    for data in trg:
        data = data.to(device)
        trg_data.append(data)
    trg_data = torch.cat(trg_data)
    print(trg_data.shape)

    trg_data = normalize(trg_data)
    print(generated.min(), generated.max(), trg_data.min(), trg_data.max())
    computed_fid = fid.calculate_fid(trg_data, generated, 512, device, 2048)
    print(f'FID: {computed_fid}')
