import os
import torch
from models.vmtc_repr.model import Classifier
from ..model import Generator, MappingNetwork, ss_model
import torchvision
from torch.utils import data
from common.util import save_image, normalize
from common.loaders.images import dataset_single
from common.evaluation import fid


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
    print(err)
    ss.eval()
    return ss


def cluster_model(cluster_path):
    cluster = Classifier(256, 5, 2048)
    state_dict = torch.load(cluster_path, map_location='cpu')
    cluster.load_state_dict(state_dict)
    cluster.eval()
    return cluster


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path of the model')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--data-root', type=str, help='Path of the data')
    parser.add_argument('--category', type=str, help='Category of FID to compute')


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

    ss = ss_model(args.ss_path).cuda()
    da = cluster_model(args.da_path).cuda()

    dataset = dataset_single(os.path.join(args.data_root, 'real' if args.domain else 'sketch', args.category))
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = dataset_single(os.path.join(args.data_root, 'sketch' if args.domain else 'real', args.category))
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    with torch.no_grad():
        print('Fetching generated data')
        d = torch.tensor(0==args.domain).repeat(batch_size).long().to(device)
        for data in src:
            data = data.to(device)
            d_trg = d[:data.shape[0]]
            y_trg = da(ss((data+1)/2)).argmax(1)
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
