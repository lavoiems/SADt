import os
import torch
from ..model import Generator, MappingNetwork, ss_model, cluster_model
import torchvision.utils as vutils
from common.loaders.images import dataset_single


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--data-root', type=str, help='Path to the data')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model-path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')


def execute(args):
    state_dict_path = args.state_dict_path
    data_root = args.data_root
    domain = args.domain
    ss_path = args.ss_path
    da_path = args.da_path
    name = args.save_name

    device = 'cuda'
    N = 5
    latent_dim = 16
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork()
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    ss = ss_model(ss_path).cuda()
    da = cluster_model(da_path).cuda()

    dataset = dataset_single(os.path.join(data_root, 'sketch' if domain else 'real', 'all'))
    idxs = [0, 15, 31, 50, 60]
    data = []
    for i in range(N):
        idx = idxs[i]
        data.append(dataset[idx])
    data = torch.stack(data).to(device)

    with torch.no_grad():
        y_src = da(ss((data+1)*0.5)).argmax(1)
        print(y_src)

    # Infer translated images
    d_trg_list = [torch.tensor(0==domain).repeat(25).long().to(device)]
    z_trg_list = torch.cat(5*[torch.randn(1, 5, latent_dim)]).to(device)
    z_trg_list = z_trg_list.transpose(0,1).reshape(25, latent_dim)
    z_trg_list = torch.stack([z_trg_list])
    data = torch.cat(5*[data])
    y_src = torch.cat(5*[y_src])
    print(z_trg_list.shape, data.shape, y_src.shape)

    N, C, H, W = data.size()
    x_concat = [data]

    for i, d_trg in enumerate(d_trg_list):

        for z_trg in z_trg_list:
            print(z_trg.shape, y_src.shape, d_trg.shape)
            s_trg = mapping(z_trg, y_src, d_trg)
            print(data.shape, s_trg.shape)
            x_fake = generator(data, s_trg)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    print(x_concat[:5].shape, x_concat[N:].shape)
    results = torch.cat([x_concat[:5], x_concat[N:]])
    save_image(results, 5, f'{name}.png')
