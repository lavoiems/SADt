import time
import copy
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models
from common.initialize import initialize, infer_iteration
from . import model


def ce_loss(data, label, dom, classifier):
    pred = classifier(data, dom)
    return F.cross_entropy(pred, label)


def cycle_loss(data, nz, generator1, generator2, device):
    z = torch.randn(data.shape[0], nz, device=device)
    gen = generator1(data, z)
    z = torch.randn(data.shape[0], nz, device=device)
    cycle = generator2(gen, z)
    return F.l1_loss(data, cycle)


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def compute_disc_loss(target, generated, domx, domy, discriminator):
    target.require_grad_()
    dt = discriminator(target, domx)
    lt = adv_loss(dt, 1)
    lgp = r1_reg(dt, target)

    dg = discriminator(generated, domy)
    lg = adv_loss(dg, 0)
    return lt, lgp, lg


def gen_mapping_loss(datax, label, domx, domy, z_dim, mapping_network, style_encoder, generator, discriminator, device):
    z = torch.randn(datax.shape[0], z_dim, device=device)
    s = mapping_network(z, label, domy)
    gen = generator(datax, s)

    # loss adv
    dg = discriminator(gen, domy)
    ladv = adv_loss(dg, 1)

    # loss class
    lcl = ce_loss(gen, label, domy, discriminator.classify)

    # loss style
    s_pred = style_encoder(gen, label, domx)
    lsty = torch.mean(torch.abs(s_pred - s))

    # loss cycle
    s_cyc = style_encoder(datax, label, domx)
    x_cyc = generator(gen, s_cyc)
    lcyc = torch.mean(torch.abs(x_cyc - datax))
    return ladv, lcl, lsty, lcyc


def gen_style_loss(datax, datay, label, domx, domy, style_encoder, generator, discriminator):
    s = style_encoder(datay, label, domy)
    gen = generator(datax, s)

    # loss adv
    dg = discriminator(gen, domy)
    ladv = adv_loss(dg, 1)

    # loss class
    lcl = ce_loss(gen, label, domy, discriminator.classify)

    # loss style
    s_pred = style_encoder(gen, label, domx)
    lsty = torch.mean(torch.abs(s_pred - s))

    # loss cycle
    s_cyc = style_encoder(datax, label, domx)
    x_cyc = generator(gen, s_cyc)
    lcyc = torch.mean(torch.abs(x_cyc - datax))
    return ladv, lcl, lsty, lcyc


def disc_mapping_loss(datax, label, domx, domy, z_dim, mapping_network, generator, discriminator, device):
    z = torch.randn(datax.shape[0], z_dim, device=device)
    s = mapping_network(z, label, domy)
    gen = generator(datax, s)
    lt, lgp, lg = compute_disc_loss(datax, gen, domx, domy, discriminator)
    lcl = ce_loss(datax, label, domx, discriminator.classify)
    return lt, lgp, lg, lcl


def disc_style_loss(datax, datay, label, domx, domy, style_encoder, generator, discriminator):
    s = style_encoder(datay, label, domy)
    gen = generator(datax, s, domy)
    lt, lgp, lg = compute_disc_loss(datax, gen, domx, domy, discriminator)
    lcl = ce_loss(datay, label, domy, discriminator.classify)
    return lt, lgp, lg, lcl


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def define_models(**parameters):
    generator = model.Generator(**parameters)
    mapping_network = model.MappingNetwork(**parameters)
    style_encoder = model.StyleEncoder(**parameters)
    discriminator = model.Discriminator(**parameters)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    return {
        'generator': generator,
        'mapping_network': mapping_network,
        'style_encoder': style_encoder,
        'discriminator': discriminator,
        'generator_ema': generator_ema,
        'mapping_network_ema': mapping_network_ema,
        'style_encoder_ema': style_encoder_ema,
    }


def train(args):
    parameters = vars(args)
    train_loader, valid_loader, test_loader = args.loaders

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    generator = models['generator'].to(args.device)
    mapping_network = models['mapping_network'].to(args.device)
    style_encoder = models['style_encoder'].to(args.device)
    discriminator = models['discriminator'].to(args.device)

    generator_ema = models['generator_ema'].to(args.device)
    mapping_network_ema = models['mapping_network_ema'].to(args.device)
    style_encoder_ema = models['style_encoder_ema'].to(args.device)

    print(generator)
    print(mapping_network)
    print(style_encoder)
    print(discriminator)

    optim_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_mapping_network = optim.Adam(mapping_network.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_style_encoder = optim.Adam(style_encoder.paramters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optims = {
        'optim_generator': optim_generator,
        'optim_mapping_network': optim_mapping_network,
        'optim_style_encoder': optim_style_encoder,
        'optim_discriminator': optim_discriminator,
    }

    iterator = iter(train_loader)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        generator.train()
        mapping_network.train()
        style_encoder.train()
        discriminator.train()

        for _ in range(args.d_updates):
            batch, iterator = sample(iterator, train_loader)
            datax = batch[0].to(args.device)
            label = batch[1].to(args.device)
            domx = batch[2].to(args.device)
            datay = batch[3].to(args.device)
            domy = batch[4].to(args.device)

            lt, lgp, lg, lcl = disc_mapping_loss(datax, label, domx, domy, args.z_dim, mapping_network, generator,
                                                 discriminator, args.device)
            dmloss = lt + lg + args.lambda_gp*lgp + args.lambda_dclass*lcl
            optim_discriminator.zero_grad()
            dmloss.backward()
            optim_discriminator.step()

            lt, lgp, lg, lcl = disc_style_loss(datax, datay, label, domx, domy, style_encoder, generator, discriminator)
            dsloss = lt + lg + args.lambda_gp*lgp + args.lambda_dclass*lcl
            optim_discriminator.zero_grad()
            dsloss.backward()
            optim_discriminator.step()

        batch, iterator = sample(iterator, train_loader)
        datax = batch[0].to(args.device)
        label = batch[1].to(args.device)
        domx = batch[2].to(args.device)
        datay = batch[3].to(args.device)
        domy = batch[4].to(args.device)

        ladv, lcl, lsty, lcyc = gen_mapping_loss(datax, label, domx, domy, args.z_dim, mapping_network, style_encoder, generator, discriminator, args.device)
        gmloss = ladv + args.lambda_lcl*lcl + args.lambda_lsty*lsty + args.lambda_lcyc*lcyc
        optim_generator.zero_grad()
        optim_mapping_network.zero_grad()
        optim_style_encoder.zero_grad()
        gmloss.backward()
        optim_style_encoder.step()
        optim_mapping_network.step()
        optim_generator.step()

        ladv, lcl, lsty, lcyc = gen_style_loss(datax, datay, label, domx, domy, style_encoder, generator, discriminator)
        gsloss = ladv + args.lambda_lcl*lcl + args.lambda_lsty*lsty + args.lambda_lcyc*lcyc
        optim_generator.zero_grad()
        gsloss.backward()
        optim_generator.step()

        moving_average(generator, generator_ema, beta=0.999)
        moving_average(mapping_network, mapping_network_ema, beta=0.999)
        moving_average(style_encoder, style_encoder_ema, beta=0.999)

        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0, end='\t')
            generator_ema.eval()
            mapping_network_ema.eval()
            style_encoder_ema.eval()
            save_path = args.save_path
            print(f'Discriminator mapping loss: {dmloss}, '
                  f'discriminator style loss: {dsloss}, '
                  f'generator mapping loss: {gmloss}, '
                  f'generator style loss: {gsloss})')
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
            save_models(optims, i, args.model_path, args.checkpoint)
