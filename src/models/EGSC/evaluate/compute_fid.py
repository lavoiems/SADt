import torch
from models.vmtc_repr.model import Classifier
from ..model import Generator, StyleEncoder
import torchvision
from common.util import save_image, normalize
from common.evaluation import fid
from torchvision.models import vgg19
from common.loaders import images


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
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--data-root-tgt', type=str, help='Path to the data')
    parser.add_argument('--dataset-src', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--dataset-trg', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--img-size', type=int, default=256, help='Size of the image')
    parser.add_argument('--bottleneeck-size', type=int, default=64, help='Size of the bottleneck')
    parser.add_argument('--bottleneck_blocks', type=int, default=4, help='Number of layers at the bottleneck')
    parser.add_argument('--category', type=str, help='Category of FID to compute')


@torch.no_grad()
def execute(args):
    device = 'cuda'
    batch_size = 128
    # Load model

    state_dict = torch.load(args.state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4).to(device)
    generator.load_state_dict(state_dict['generator'])
    style_encoder = StyleEncoder(img_size=args.img_size).to(device)
    style_encoder.load_state_dict(state_dict['style_encoder'])

    feature_blocks = 29 if args.img_size == 256 else 8
    vgg = vgg19(pretrained=True).features[:feature_blocks].to(device)

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(args.data_root_src)
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = getattr(images, args.dataset_trg)
    trg_dataset = dataset(args.data_root_tgt)

    print(f'Src size: {len(src_dataset)}, Tgt size: {len(trg_dataset)}')
    generated = []
    print('Fetching generated data')
    d = torch.tensor(args.domain).repeat(batch_size).long().to(device)
    for data in src:
        data = data.to(device)
        d_trg = d[:data.shape[0]]
        features = vgg(data)
        for i in range(10):
            x_idxs = torch.randint(low=0, high=len(trg_dataset), size=(len(data),))
            x_trg = torch.stack([trg_dataset[idx].to(device) for idx in x_idxs])
            x_trg = x_trg * 2 - 1
            s_trg = style_encoder(x_trg, d_trg)
            gen = generator(data, features, s_trg)
            generated.append(gen)
    generated = torch.cat(generated)
    generated = normalize(generated)
    save_image(generated[:4], 'Debug.png')

    print('Fetching target data')
    trg_data = []
    for data in trg_dataset:
        data = data.to(device)
        trg_data.append(data)
    trg_data = torch.cat(trg_data)
    print(trg_data.shape)

    trg_data = normalize(trg_data)
    print(generated.min(), generated.max(), trg_data.min(), trg_data.max())
    computed_fid = fid.calculate_fid(trg_data, generated, 512, device, 2048)
    print(f'FID: {computed_fid}')
