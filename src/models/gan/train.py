import time
import torch
from torch import optim

from common.util import sample, save_models
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


def disc_loss_generation(data, nz, discriminator, generator1, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator1(z).detach()
    pos_loss = discriminator(data).mean()
    neg_loss = discriminator(gen).mean()
    gp = gp_loss(gen, data, discriminator, device)
    return pos_loss, neg_loss, gp


def transfer_loss(batch_size, nz, discriminator, generator, device):
    z = torch.randn(batch_size, nz, device=device)
    gen = generator(z)
    loss = discriminator(gen).mean()
    return loss


def define_models(shape1, **parameters):
    discriminator = model.Discriminator(shape1[0], shape1[1], **parameters)
    generator = model.Generator(shape1[0], shape1[1], **parameters)
    return {
        'generator': generator,
        'discriminator': discriminator,
    }


@torch.no_grad()
def evaluate(visualiser, noise, data, transfer, id):
    transfered = transfer(noise).to('cpu').detach().numpy()
    visualiser.image(transfered, title=f'GAN generated', step=id)
    visualiser.image(data.cpu().numpy(), title=f'Target', step=id)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    discriminator = models['discriminator'].to(args.device)
    print(generator)
    print(discriminator)

    optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1, iter2 = iter(train_loader1), iter(train_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        discriminator.train()
        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            datax = batchx[0].to(args.device)
            batchy, iter2 = sample(iter2, train_loader2)
            datay = batchy[0].to(args.device)
            data = torch.cat((datax, datay))
            optim_discriminator.zero_grad()
            d_pos_loss, d_neg_loss, gp = disc_loss_generation(data, args.z_dim, discriminator, generator, args.device)
            d_pos_loss.backward()
            d_neg_loss.backward(mone)
            (10*gp).backward()
            optim_discriminator.step()

        optim_generator.zero_grad()
        t_loss = transfer_loss(args.train_batch_size, args.z_dim, discriminator, generator, args.device)
        t_loss.backward()
        optim_generator.step()

        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0)
            noise = torch.randn(data.shape[0], args.z_dim, device=args.device)
            evaluate(args.visualiser, noise, data, generator, i)
            d_loss = (d_pos_loss-d_neg_loss).detach().cpu().numpy()
            args.visualiser.plot(step=i, data=d_loss, title=f'Discriminator loss')
            args.visualiser.plot(step=i, data=t_loss.detach().cpu().numpy(), title=f'Generator loss')
            args.visualiser.plot(step=i, data=gp.detach().cpu().numpy(), title=f'Gradient Penalty')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
