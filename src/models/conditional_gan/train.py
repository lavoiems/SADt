import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models, one_hot_embedding
from common.initialize import initialize, infer_iteration
from . import model


def gp_loss(x, y, label, d, device):
    batch_size = x.size()[0]
    gp_alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    interp = gp_alpha * x.data + (1 - gp_alpha) * y.data
    interp.requires_grad = True
    label.requires_grad = True
    d_interp = d(interp, label)
    grad_interp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                      grad_outputs=torch.ones(d_interp.size(), device=device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_interp = grad_interp.view(grad_interp.size(0), -1)
    diff = grad_interp.norm(2, dim=1) - 1
    diff = torch.clamp(diff, 0)
    return torch.mean(diff**2)


def critic_loss(data, label, nz, critic, generator1, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator1(z, label).detach()
    pos_loss = critic(data, label).mean()
    neg_loss = critic(gen, label).mean()
    gp = gp_loss(gen, data, label, critic, device)
    return pos_loss, neg_loss, gp


def transfer_loss(batch_size, label, nz, critic, generator, device):
    z = torch.randn(batch_size, nz, device=device)
    gen = generator(z, label)
    loss = critic(gen, label).mean()
    return loss


def define_models(shape1, **parameters):
    critic = model.Critic(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'critic': critic,
    }


def corrupt(label, nc, rate):
    if rate:
        N = int(len(label) * rate)
        idxes = np.random.choice(len(label), N, replace=False)
        label[idxes] = np.random.choice(nc, len(idxes))
    return label


@torch.no_grad()
def plot_transfer(visualiser, label, nc, nz, data, target, generator, device, i):
    z = torch.randn(target.shape[0], nz, device=device)
    visualiser.image(data.cpu().numpy(), 'source', i)
    visualiser.image(target.cpu().numpy(), 'target', i)
    X = generator(z, label)
    visualiser.image(X.cpu().numpy(), f'Generated', i)

    merged = len(X)*2 * [None]
    merged[:2*len(data):2] = data
    merged[1:2*len(X):2] = X
    merged = torch.stack(merged)
    visualiser.image(merged.cpu().numpy(), f'Comparison', i)

    z = torch.stack(nc*[z[:nc-1]]).transpose(0, 1).reshape(-1, z.shape[1])
    data1 = torch.cat((nc-1)*[data[:nc]])
    X = generator(z, label)
    X = torch.cat((data1[:nc], X))
    visualiser.image(X.cpu().numpy(), f'Z effect', i)


@torch.no_grad()
def evaluate(loader, nz, nc, rate, transfer, classifier, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        clabel = corrupt(label, nc, rate)
        z = torch.randn(data.shape[0], nz, device=device)
        gen = transfer(z, clabel)
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    return accuracy


def train(args):
    parameters = vars(args)
    valid_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    critic = models['critic'].to(args.device)
    eval = args.evaluation.eval().to(args.device)
    print(generator)
    print(critic)

    optim_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter2 = iter(train_loader2)
    titer, titer2 = iter(test_loader1), iter(test_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        critic.train()
        for _ in range(args.d_updates):
            batch, iter2 = sample(iter2, train_loader2)
            data = batch[0].to(args.device)
            label = corrupt(batch[1], args.nc, args.corrupt_tgt)
            label = one_hot_embedding(label, args.nc).to(args.device)
            optim_critic.zero_grad()
            pos_loss, neg_loss, gp = critic_loss(data, label, args.z_dim, critic, generator, args.device)
            pos_loss.backward()
            neg_loss.backward(mone)
            (10*gp).backward()
            optim_critic.step()

        optim_generator.zero_grad()
        t_loss = transfer_loss(data.shape[0], label, args.z_dim, critic, generator, args.device)
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0)
            generator.eval()
            batch, titer = sample(titer, test_loader1)
            data1 = batch[0].to(args.device)
            label = one_hot_embedding(batch[1], args.nc).to(args.device)
            batch, titer = sample(titer2, test_loader2)
            data2 = batch[0].to(args.device)
            plot_transfer(args.visualiser, label, args.nc, data1, data2, args.nz, generator, args.device, i)
            save_path = args.save_path
            eval_accuracy = evaluate(valid_loader1, args.nz, args.nc, args.corrupt_src, generator, eval, args.device)
            test_accuracy = evaluate(test_loader1, args.nz, args.nc, args.corrupt_src, generator, eval, args.device)
            with open(os.path.join(save_path, 'critic_loss'), 'a') as f: f.write(f'{i},{(pos_loss-neg_loss).cpu().item()}\n')
            with open(os.path.join(save_path, 'tloss'), 'a') as f: f.write(f'{i},{t_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'eval_accuracy'), 'a') as f: f.write(f'{i},{eval_accuracy}\n')
            with open(os.path.join(save_path, 'test_accuracy'), 'a') as f: f.write(f'{i},{eval_accuracy}\n')
            args.visualiser.plot((pos_loss-neg_loss).cpu().detach().numpy(), title='critic_loss', step=i)
            args.visualiser.plot(t_loss.cpu().detach().numpy(), title='tloss', step=i)
            args.visualiser.plot(eval_accuracy, title=f'Validation transfer accuracy', step=i)
            args.visualiser.plot(test_accuracy, title=f'Test transfer accuracy', step=i)

            t0 = time.time()
            save_models(models, 0, args.model_path, args.checkpoint)
