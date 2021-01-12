# -*- coding: utf-8 -*-
"""
Scripts used to train the unsupervised model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import utils.helper as helper
from utils.losses import VaeLoss
from models.lmf import LMF
from utils.muad_dataset import MuadDataset
from utils.parser import train_parser


def train(args):
    """
    Train function
    :param args:
    :return:
    """
    # load dataset first #TODO: Use args or config file for data path
    print("loading dataset")
    dataset = MuadDataset("data/CrisisMMD_v2.0/CrisisMMD_v2.0",
                          "data/CrisisMMD_v2.0/train_cleaned.xlsx")
    torch.multiprocessing.set_start_method('spawn')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print("creating model")
    model = LMF(rank=args.rank, latent_dim=args.latent_dim)
    if args.use_gpu:
        model.cuda()

    # define loss
    loss_func = VaeLoss()
    # define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    if args.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = helper.load_saved_model(saved_path, model)
    else:
        # setup saved model folder
        init_epoch = 0
        saved_path = helper.setup_train(args)

    # used to record training curve
    writer = SummaryWriter(saved_path)
    print('training start')
    step = 0

    for epoch in range(init_epoch, args.epoches):
        scheduler.step(epoch)
        for i, sample_batched in enumerate(dataloader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            image_features, text_features, label = sample_batched['image_features'], sample_batched['text_features'], \
                                                   sample_batched['label']

            if args.use_gpu:
                image_features = image_features.cuda()
                text_features = text_features.cuda()
                label = label.cuda()

            output = model(image_features, text_features)
            loss = loss_func(output)

            # back-propagation
            loss['total_loss'].backward()
            optimizer.step()

            # evaluation for every n step #TODO: Make this parameterized
            if step % 1 == 0:
                print("[epoch %d][%d/%d], total loss: %.4f" % (epoch + 1, i + 1, len(dataloader),
                                                               loss['total_loss'].item()))
                writer.add_scalar('total loss', loss['total_loss'].item(), step)
                writer.add_scalar('recons loss', loss['reconstruction_loss'].item(), step)
                writer.add_scalar('kl_loss', loss['kl_loss'].item(), step)

            step += 1

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))


if __name__ == '__main__':
    opt = train_parser()
    train(opt)
