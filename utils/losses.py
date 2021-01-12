# -*- coding: utf-8 -*-
"""
Define different kinds of loss
"""

import numpy as np
import torch
import torch.nn as nn


class VaeLoss(nn.Module):
    """
    Vae_loss = Reconstruction_loss + KL_loss
    """

    def __init__(self, gamma=1.0, capacity=0.0):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, output):
        """
        Caclulate the loss in the forwarding
        :param output: a dictionary containing all the info we need for loss
        :return:
        """
        batch_size, n_features = output['reconstruct_input'].shape
        origin_input = torch.cat([output['normalized_image_input'],
                                  output['normalized_text_input']],
                                 dim=1)
        reconstruction_loss = self.reconstruction_loss(output['reconstruct_input'],
                                                       origin_input)
        # Kl loss
        z_var = output['var']
        z_mu = output['mean']
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)

        return {'total_loss': (0.1 * kl_loss + reconstruction_loss) / batch_size,
                'kl_loss': kl_loss,
                'reconstruction_loss': reconstruction_loss}
