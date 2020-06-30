import torch
from PIL import Image
import os
from src.models.i2i.model import Generator, MappingNetwork
from src.models.vmtc_repr.model import Classifier
import sys
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
  def __init__(self, dataroot, setname):
    self.dataroot = dataroot
    images = os.listdir(os.path.join(self.dataroot, 'test' + setname))
    self.img = [os.path.join(self.dataroot, 'test' + setname, x) for x in images]
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


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


if __name__ == '__main__':
    state_dict_path, data_root, domain = sys.argv[1], sys.argv[2], sys.argv[3]
    device = 'cuda'
    latent_dim = 16
    domain = int(domain)
    d1_nsamples = 3175
    batch_size = 128
    ss_path = '/network/tmp1/lavoiems/moco_v2_800ep_pretrain.pth.tar'
    da_path = '/network/tmp1/lavoiems/vmtc_repr/vmtc-repr_ln-sketch-real-None/model/classifier_13029'
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(w_hpf=0).to(device).eval()
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork()
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device).eval()

    ss = ss_model(ss_path).cuda()
    da = cluster_model(da_path).cuda()

    dataset = dataset_single(data_root, 'A' if domain else 'B')
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = dataset_single(data_root, 'B' if domain else 'A')
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    with torch.no_grad():
        print('Fetching generated data')
        d = torch.tensor(0==domain).repeat(batch_size).long().to(device)
        for data in src:
            data = data.to(device)
            maps = torch.LongTensor([3, 0, 4, 2, 1]).to(device)
            d_trg = d[:data.shape[0]]
            y_trg = da(ss(data+1)/2).argmax(1)
            y_trg = maps[y_trg]
            for i in range(10):
                z_trg = torch.randn(data.shape[0], latent_dim, device=device)
                s_trg = mapping(z_trg, y_trg, d_trg)
                print(data.shape, s_trg.shape)
                gen = generator(data, s_trg, masks=None)
                generated.append(gen)
        generated = torch.cat(generated)
        save_image(generated[:4], 4, 'Debug.png')

        print('Fetching target data')
        trg_data = []
        for data in trg:
            data = data.to(device)
            print(data.shape)
            trg_data.append(data)
        trg_data = torch.cat(trg_data)
        print(trg_data.shape)

        fid = fid.calculate_fid(trg_data, generated.cpu(), 512, device, 2048)
    print(f'FID: {fid}')
