from importlib import import_module
from common.util import get_args
from common.initialize import load_last_model
import torch
import argparse
from PIL import Image
import os
from models.vmtc_repr.model import Classifier
import torchvision
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torch.utils import data
import torchvision.utils as vutils


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
  def __init__(self, dataroot, setname):
    self.dataroot = dataroot
    categories = os.listdir(os.path.join(self.dataroot, 'test' + setname))
    self.img = []
    for cat in categories:
        images = os.listdir(os.path.join(self.dataroot, 'test'+setname, cat))
        self.img += [os.path.join(self.dataroot, 'test' + setname, cat, x) for x in images]
    self.img = list(sorted(self.img))
    self.size = len(self.img)
    self.input_dim = 3 

    # setup image transformation
    transforms = [Resize((256, 256), 1)] 
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

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
    return self.size

#class dataset_single(data.Dataset):
#  def __init__(self, dataroot, setname):
#    self.dataroot = dataroot
#    categories = os.listdir(os.path.join(self.dataroot, setname, 'test'))
#    self.img = []
#    for cat in categories:
#        images = os.listdir(os.path.join(self.dataroot, setname, 'test', cat))
#        self.img += [os.path.join(self.dataroot, setname, 'test', cat, x) for x in images]
#    self.img = list(sorted(self.img))
#    self.size = len(self.img)
#    self.input_dim = 3
#
#    # setup image transformation
#    transforms = [Resize((256, 256), 1)]
#    transforms.append(ToTensor())
#    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
#    self.transforms = Compose(transforms)
#    print('%s: %d images'%(setname, self.size))
#    return
#
#  def __getitem__(self, index):
#    data = self.load_img(self.img[index], self.input_dim)
#    return data
#
#  def load_img(self, img_name, input_dim):
#    img = Image.open(img_name).convert('RGB')
#    img = self.transforms(img)
#    if input_dim == 1:
#      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
#      img = img.unsqueeze(0)
#    return img
#
#  def __len__(self):
#    return self.size


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='Path of the model directory')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--data-root', type=str, help='Path of the data')
    args = parser.parse_args()

    device = 'cuda'
    N = 5
    latent_dim = 16
    d1_nsamples = 3175
    # Load model
    model_definition = import_module('.'.join(('models', args.model, 'train')))
    model_parameters = get_args(args.model_path)
    print(model_parameters)
    models = model_definition.define_models(**model_parameters)
    generator = models['generator_ema']
    generator = load_last_model(generator, 'generator_ema', args.model_path).to(device)
    mapping = models['mapping_network_ema']
    mapping = load_last_model(mapping, 'mapping_network_ema', args.model_path).to(device)

    ss = ss_model(args.ss_path).cuda()
    da = cluster_model(args.da_path).cuda()

    dataset = dataset_single(args.data_root, 'A' if args.domain else 'B')
    idxs = [0, 15, 35, 50, 60]
    data = []
    for i in range(N):
        idx = idxs[i]
        data.append(dataset[idx])
    data = torch.stack(data).to(device)

    with torch.no_grad():
        y_src = da(ss((data+1)*0.5)).argmax(1)

    # Infer translated images
    d_trg_list = [torch.tensor(0==args.domain).repeat(25).long().to(device)]
    z_trg_list = torch.cat(5*[torch.randn(1, 5, latent_dim)]).to(device)
    z_trg_list = z_trg_list.transpose(0,1).reshape(25, latent_dim)
    z_trg_list = torch.stack([z_trg_list])
    data = torch.cat(5*[data])
    y_src = torch.cat(5*[y_src])
    print(z_trg_list.shape, data.shape, y_src.shape)

    N, C, H, W = data.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [data]

    for i, d_trg in enumerate(d_trg_list):
        for z_trg in z_trg_list:
            print(z_trg.shape, y_src.shape, d_trg.shape)
            s_trg = mapping(z_trg, y_src, d_trg)
            print(data.shape, s_trg.shape)
            x_fake = generator(data, s_trg)
            x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    results = [None] * len(x_concat)
    print(x_concat[:5].shape, x_concat[N:].shape)
    results = torch.cat([x_concat[:5], x_concat[N:]])
    save_image(results, 5, f'samples_name:{args.model}_domain:{args.domain}.png')
