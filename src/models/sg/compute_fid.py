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
from evaluation import fid


class dataset_single(data.Dataset):
  def __init__(self, dataroot, setname, category):
    self.dataroot = dataroot
    images = os.listdir(os.path.join(self.dataroot, setname, 'fid', category))
    self.img = [os.path.join(self.dataroot, setname, 'fid', category, x) for x in images]
    self.img = list(sorted(self.img))
    self.size = len(self.img)
    self.input_dim = 3

    # setup image transformation
    transforms = [Resize((256, 256), 1)]
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
      return len(self.img)


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--data-root', type=str, help='Path to the data')
    parser.add_argument('--category', type=str, help='Category of FID to compute')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_root = args.data_root

    device = 'cuda'
    batch_size = 128
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=256, max_conv_dim=512).to(device)
    generator.load_state_dict(state_dict['generator'])
    style = StyleEncoder(img_size=256, max_conv_dim=512)
    style.load_state_dict(state_dict['style_encoder'])
    style.to(device)

    dataset = dataset_single(args.data_root, 'sketch' if args.domain else 'real', args.category)
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = dataset_single(args.data_root, 'real' if args.domain else 'sketch', args.category)
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)

    features = vgg19(pretrained=True).features[:29]
    features = features.to(device)
    features.eval()

    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    iter_tgt = iter(trg)
    with torch.no_grad():
        print('Fetching generated data')
        d = torch.tensor(0==args.domain).repeat(batch_size).long().to(device)
        for data in src:
            data = data.to(device)
            d_trg = d[:data.shape[0]]
            examplar = next(iter_tgt)
            examplar = examplar[:data.shape[0]].to(device)
            print(examplar.shape, d_trg.shape, data.shape)
            s_trg = style(examplar, d_trg)
            f = features(data)
            gen = generator(data, f, s_trg)
            generated.append(gen)
        generated = torch.cat(generated)
        save_image(generated[:4], 4, 'Debug.png')

        print('Fetching target data')
        trg_data = []
        for data in trg:
            data = data.to(device)
            trg_data.append(data)
        trg_data = torch.cat(trg_data)
        print(trg_data.shape)

        fid = fid.calculate_fid(trg_data, generated, 512, device, 2048)
    print(f'FID: {fid}')
