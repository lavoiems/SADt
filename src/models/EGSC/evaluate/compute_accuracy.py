import torch
from ..model import Generator, StyleEncoder
import torchvision.utils as vutils
from common.loaders import images
from torchvision.models import vgg19
import torch.nn.functional as F
from common.initialize import define_last_model


@torch.no_grad()
def evaluate(loader, trg_dataset, domain, style_encoder, vgg, generator, classifier, device):
    correct = 0
    total = 0

    for data, label in loader:
        N = len(data)
        d_trg = torch.tensor(0 == domain).repeat(N).long().to(device)
        x_idxs = torch.randint(low=0, high=len(trg_dataset), size=(N,))
        x_trg = torch.stack([trg_dataset[idx][0].to(device) for idx in x_idxs])
        x_trg = x_trg * 2 - 1
        data, label = data.to(device), label.to(device)
        data = data*2 - 1

        s = style_encoder(x_trg, d_trg)
        f = vgg(data)
        gen = generator(data, f, s)
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

    classifier = define_last_model('classifier', args.classifier_path, 'classifier', shape=3, nc=10).to(device)
    classifier.eval()

    feature_blocks = 29 if args.img_size == 256 else 8
    vgg = vgg19(pretrained=True).features[:feature_blocks].to(device)

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(data_root_src, 1, 1)[2]
    dataset = getattr(images, args.dataset_trg)
    trg_dataset = dataset(data_root_tgt, 1, 1)[2].dataset

    accuracy = evaluate(src_dataset, trg_dataset, domain, style_encoder, vgg, generator, classifier, device)
    print(accuracy)

