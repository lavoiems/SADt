from importlib import import_module
import torch
from PIL import Image
import os
from models.vmtc_repr.model import Classifier
import argparse
import torchvision
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torch.utils import data
import torchvision.utils as vutils
from evaluation import fid


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
    parser.add_argument('--model-path', type=str, help='Path of the model')
    parser.add_argument('--model', type=str, help='Model name', default='sg_sem')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--data-root', type=str, help='Path of the data')
    parser.add_argument('--category', type=str, help='Category of FID to compute')
    args = parser.parse_args()

    device = 'cuda'
    latent_dim = 16
    d1_nsamples = 3175
    batch_size = 128
    # Load model
    state_dict = torch.load(args.model_path, map_location='cpu')
    model_definition = import_module('.'.join(('models', args.model, 'model')))

    generator = model_definition.Generator(bottleneck_size=64, bottleneck_blocks=4).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = model_definition.MappingNetwork()
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    ss = ss_model(args.ss_path).cuda()
    da = cluster_model(args.da_path).cuda()

    dataset = dataset_single(args.data_root, 'sketch' if args.domain else 'real', args.category)
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = dataset_single(args.data_root, 'real' if args.domain else 'sketch', args.category)
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    with torch.no_grad():
        print('Fetching generated data')
        d = torch.tensor(0==args.domain).repeat(batch_size).long().to(device)
        for data in src:
            data = data.to(device)
            d_trg = d[:data.shape[0]]
            y_trg = da(ss(data+1)/2).argmax(1)
            for i in range(10):
                z_trg = torch.randn(data.shape[0], latent_dim, device=device)
                s_trg = mapping(z_trg, y_trg, d_trg)
                gen = generator(data, s_trg)
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

        trg_data = (trg_data + 1) / 2
        generated = (generated + 1) / 2
        fid = fid.calculate_fid(trg_data, generated, 512, device, 2048)
    print(f'FID: {fid}')
