# -*- coding: utf-8 -*-
"""
Helper function for training and testing
"""
import os
import glob
import re

from datetime import datetime

import torch


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted
    :param saved_path:  model saved path, str
    :param model:  model object
    :return:
    """
    if not os.path.exists(saved_path):
        raise ValueError('{} not found'.format(saved_path))

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)

    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch)))

    return initial_epoch, model


def setup_train(args):
    """
    create folder for saved model based on current timestep and model name
    :param args:
    :return:
    """
    # TODO: make this parameterized
    model_name = 'LMF'
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')
    full_path = os.path.join(current_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path
