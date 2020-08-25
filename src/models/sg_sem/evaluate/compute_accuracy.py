import torch
from ..model import Generator, MappingNetwork, semantics
import torchvision.utils as vutils
from common.loaders import images
import torch.nn.functional as F
from common.initialize import define_last_model


def evaluate(loader, nz, domain, sem, mapping, generator, classifier, device):
    correct = 0
    total = 0

    for data, label in loader:
        N = len(data)
        d_trg = torch.tensor(domain).repeat(N).long().to(device)
        data, label = data.to(device), label.to(device)
        data = data*2 - 1

        y = sem(data)
        z = torch.randn(N, nz).to(device)
        s = mapping(z, y, d_trg)
        gen = generator(data, s)

        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    return accuracy


def save_image(x, ncol, filename):
    print(x.min(), x.max())
    x.clamp_(-1, 1)
    x = (x + 1) / 2
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=2, pad_value=1)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--classifier-path', type=str, help='Path to the classifier model')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--dataset-src', type=str, default='dataset_single', help='name of the dataset')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--max-conv-dim', type=int, default=128)
    parser.add_argument('--bottleneeck-size', type=int, default=64, help='Size of the bottleneck')
    parser.add_argument('--bottleneck_blocks', type=int, default=4, help='Number of layers at the bottleneck')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')


@torch.no_grad()
def execute(args):
    state_dict_path = args.state_dict_path
    data_root_src = args.data_root_src
    domain = args.domain
    nz = 16

    device = 'cuda'
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size, max_conv_dim=args.max_conv_dim).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork(nc=10)
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    sem = semantics(None, 'vmt_cluster', args.da_path).cuda()

    classifier = define_last_model('classifier', args.classifier_path, 'classifier', shape=3, nc=10).to(device)
    classifier.eval()

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(data_root_src, 1, 1)[2]

    accuracy = evaluate(src_dataset, nz, domain, sem, mapping, generator, classifier, device)
    print(accuracy)

