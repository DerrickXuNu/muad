# -*- coding: utf-8 -*-
"""
Argparser for MUAD
"""

import argparse


def train_parser():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("--dataset", type=list,
                        help='give the image path and excel path. Eg. [data/image, data/excel]')
    parser.add_argument('--use_gpu', default=True, action='store_false', help='whether using gpu or not')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument('--epoches', default=100, type=int, help='Number of iterations for training')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--save_freq', default=5,  type=int, help='model save frequency')

    # model related
    parser.add_argument('--rank', default=4, type=int, help='model save frequency')
    parser.add_argument('--latent_dim', default=20, type=int, help='model save frequency')

    opt = parser.parse_args()

    return opt
