import time
import os
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models
#from evaluation.fid import calculate_fid
from common.initialize import initialize, infer_iteration
from . import model


def gp_loss(x, y, d, device):
    batch_size = x.size()[0]
    gp_alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    interp = gp_alpha * x.data + (1 - gp_alpha) * y.data
    interp.requires_grad = True
    d_interp = d(interp)
    grad_interp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                      grad_outputs=torch.ones(d_interp.size(), device=device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_interp = grad_interp.view(grad_interp.size(0), -1)
    diff = grad_interp.norm(2, dim=1) - 1
    diff = torch.clamp(diff, 0)
    return torch.mean(diff**2)


def semantic_loss(data, nz, generator, classifier, device):
    label = classifier(data).argmax(1)
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(data, z)
    pred = classifier(gen)
    return F.cross_entropy(pred, label)


def cycle_loss(data, nz, generator1, generator2, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator1(data, z)
    z = torch.randn(data.shape[0], nz, device=device)
    cycle = generator2(gen, z)
    return F.l1_loss(data, cycle)


def identity_loss(data, nz, generator, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(data, z)
    return F.l1_loss(gen, data)


def compute_critic_loss(data, nz, target, critic, generator, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(data, z).detach()
    pos_loss = critic(target).mean()
    neg_loss = critic(gen).mean()
    gp = gp_loss(gen, target, critic, device)
    return pos_loss - neg_loss + 10*gp


def generator_loss(data, nz, critic, generator, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator(data, z)
    return critic(gen).mean()


def define_models(shape1, **parameters):
    critic = model.Critic(shape1[0], **parameters)
    generator = model.Generator(shape1[0], **parameters)
    return {
        'critic': critic,
        'generator': generator,
    }


@torch.no_grad()
def evaluate(loader, nz, transfer, classifier, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        z = torch.randn(data.shape[0], nz, device=device)
        gen = transfer(data, z)
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    return accuracy


@torch.no_grad()
def plot_transfer(visualiser, data, target, nz, transfer, id, i, device):
    z = torch.randn(data.shape[0], nz, device=device)
    transfered = transfer(data, z)
    merged = len(data)*2 * [None]
    merged[:2*len(data):2] = data
    merged[1:2*len(transfered):2] = transfered
    merged = torch.stack(merged)
    visualiser.image(merged.cpu().numpy(), f'Comparison{id}', i)
    visualiser.image(target.cpu().numpy(), title=f'Target {id}', step=i)


def train(args):
    parameters = vars(args)
    train_loader1, valid_loader1, test_loader1 = args.loaders1
    train_loader2, valid_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    critic = models['critic'].to(args.device)
    generator = models['generator'].to(args.device)
    evalY = args.evalY.to(args.device).eval()
    semantic = args.semantic.to(args.device).eval()
    print(generator)
    print(critic)
    print(semantic)

    optim_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        critic.train()
        generator.train()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            datax = batchx[0].to(args.device)

            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy[0].to(args.device)

            critic_lossy = compute_critic_loss(datax, args.z_dim, datay, critic, generator, args.device)
            optim_critic.zero_grad()
            critic_lossy.backward()
            optim_critic.step()

        batchx, iter1 = sample(iter1, train_loader1)
        datax = batchx[0].to(args.device)

        batchy, iter2 = sample(iter2, train_loader2)
        datay = batchy[0].to(args.device)

        glossxy = generator_loss(datax, args.z_dim, critic, generator, args.device)
        slossxy = semantic_loss(datax, args.z_dim, generator, semantic, args.device)
        optim_generator.zero_grad()
        glossxy.backward()
        (args.gsxy*slossxy).backward()
        optim_generator.step()


        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0)
            generator.eval()
            save_path = args.save_path
            plot_transfer(args.visualiser, datax, datay, args.z_dim, generator, 'x-y', i, args.device)
            test_accuracy_xy = evaluate(test_loader1, args.z_dim, generator, evalY, args.device)
            with open(os.path.join(save_path, 'glossxy'), 'a') as f: f.write(f'{i},{glossxy.cpu().item()}\n')
            with open(os.path.join(save_path, 'slossxy'), 'a') as f: f.write(f'{i},{slossxy.cpu().item()}\n')
            with open(os.path.join(save_path, 'test_accuracy_xy'), 'a') as f: f.write(f'{i},{test_accuracy_xy}\n')
            args.visualiser.plot(critic_lossy.cpu().detach().numpy(), title='critic_lossy', step=i)
            args.visualiser.plot(glossxy.cpu().detach().numpy(), title='glossxy', step=i)
            args.visualiser.plot(slossxy.cpu().detach().numpy(), title='slossxy', step=i)
            args.visualiser.plot(test_accuracy_xy, title=f'Test transfer accuracy X-Y', step=i)
            t0 = time.time()
            save_models(models, 0, args.model_path, args.checkpoint)


@torch.no_grad()
def evaluate_fid(args):
    parameters = vars(args)
    _, _, test_loader1 = args.loaders1
    _, _, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, True, args.save_path, args.model_path)

    generatorXY = models['generatorXY'].to(args.device)
    generatorYX = models['generatorYX'].to(args.device)

    datas1 = []
    labels1 = []
    gens1 = []
    for i, (data, label) in enumerate(test_loader1):
        data, label = data.to(args.device), label.to(args.device)
        datas1 += [data]
        labels1 += [label]
        z = torch.randn(len(data), args.z_dim, device=args.device)
        gen = generatorXY(data, z)
        gens1 += [gen]
    datas1 = torch.cat(datas1)
    labels1 = torch.cat(labels1)
    gens1 = torch.cat(gens1)

    datas2 = []
    labels2 = []
    gens2 = []
    for i, (data, label) in enumerate(test_loader2):
        data, label = data.to(args.device), label.to(args.device)
        datas2 += [data]
        labels2 += [label]
        z = torch.randn(len(data), args.z_dim, device=args.device)
        gen = generatorYX(data, z)
        gens2 += [gen]
    datas2 = torch.cat(datas2)
    labels2 = torch.cat(labels2)
    gens2 = torch.cat(gens2)

    #fid = calculate_fid(datas1[:1000], datas1[1000:2000], 50, args.device, 2048)
    #print(f'fid datasetX: {fid}')
    #fid = calculate_fid(datas2[:1000], datas2[1000:2000], 50, args.device, 2048)
    #print(f'fid datasetY: {fid}')
    fid = calculate_fid(datas1, gens2, 50, args.device, 2048)
    save_path = args.save_path
    with open(os.path.join(save_path, 'fid_yx'), 'w') as f: f.write(f'{fid}\n')
    print(f'fid Y->X: {fid}')
    fid = calculate_fid(datas2, gens1, 50, args.device, 2048)
    with open(os.path.join(save_path, 'fid_xy'), 'w') as f: f.write(f'{fid}\n')
    print(f'fid X->Y: {fid}')

    #for i in range(10):
    #    l1 = labels1 == i
    #    l2 = labels2 == i
    #    d, g = datas1[l1], gens2[l2]
    #    fid = calculate_fid(d, g, 50, args.device, 2048)
    #    print(f'intra-fid label {i} Y->X: {fid}')

    #    d, g = datas2[l2], gens1[l1]
    #    fid = calculate_fid(d, g, 50, args.device, 2048)
    #    print(f'intra-fid label {i} X->Y: {fid}')
