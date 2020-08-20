import argparse
import torch
from PIL import Image
import os
from models.EGSC.model import Generator, StyleEncoder
import sys
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torch.utils import data
import torchvision.utils as vutils
from common.loaders.images import mnist, svhn
from torchvision.models import vgg19
from models.classifier.model import Classifier
import torch.nn.functional as F


@torch.no_grad()
def evaluate(loader_src, loader_tgt, transfer, style, features, classifier, device):
    correct = 0
    total = 0
    iter_tgt = iter(loader_tgt)
    for  data, label in loader_src:
        examplar, _ = next(iter_tgt)
        data = data*2-1
        examplar = examplar*2-1
        examplar = examplar[:len(data)].to(device)
        d = torch.LongTensor([1] * len(data))
        data, label = data.to(device), label.to(device)
        s = style(examplar, d)
        f = features(data)
        gen = transfer(data, f, s)
        gen = gen.clamp_(-1, 1)
        gen = (gen+1)*0.5
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total

    vutils.save_image(data.cpu(), 'source.png')
    vutils.save_image(gen.cpu(),  'generated.png')
    return accuracy.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--data-root', type=str, help='Path to the data')
    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_root = args.data_root

    device = 'cuda'
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=32, max_conv_dim=128).to(device)
    generator.load_state_dict(state_dict['generator'])
    style = StyleEncoder(img_size=32, max_conv_dim=512)
    style.load_state_dict(state_dict['style_encoder'])
    style.to(device)

    classifier = Classifier(nc=10, depth=16, drop_rate=0.4, widen_factor=8)
    classifier_state_dict = torch.load('/network/tmp1/lavoiems/experiments/classifier/classifier-wide_svhn-None/model/classifier_45000')
    classifier.load_state_dict(classifier_state_dict)
    classifier.eval()
    classifier = classifier.to(device)

    _, _, loader_src, _, _ = mnist(data_root, 256, 256)
    _, _, loader_tgt, _, _ = svhn(data_root, 256, 256)

    features = vgg19(pretrained=True).features[:8]
    features = features.to(device)
    features.eval()
    accuracy = evaluate(loader_src, loader_tgt, generator, style, features, classifier, device)
    print(accuracy)
