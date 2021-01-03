# -*- coding: utf-8 -*-
"""
Low-rank Multi-modal Fusion, origin implementation: https://github.com/Justin1904/Low-rank-Multimodal-Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal


class LMF(nn.Module):
    """Low-Rank Multimodal Fusion
      Parameters
      ----------

      Attributes
      ----------
    """

    def __init__(self, input_dim=512, rank=4, drop_prob=0.2,
                 encoder_neurons=None,
                 decoder_neurons=None):
        super(LMF, self).__init__()
        # --------------------- LMF specifically related parameters ---------------------#
        if encoder_neurons is None:
            encoder_neurons = [720, 512, 256]

        if decoder_neurons is None:
            decoder_neurons = [256, 512, 720]

        self.rank = rank
        self.input_dim = input_dim
        self.fused_dim = encoder_neurons[0]

        self.text_factor = Parameter(torch.Tensor(self.rank, self.input_dim + 1, self.fused_dim))
        self.image_factor = Parameter(torch.Tensor(self.rank, self.input_dim + 1, self.fused_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

        xavier_normal(self.image_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        # --------------------- VAE specifically related parameters ---------------------#
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons

    def forward(self, image_feature, text_feature):
        """
        Forwarding
        :param image_feature:
        :param text_feature:
        :return:
        """
        # fuse input data by LMF
        batch_size = image_feature.data.shape[0]

        if image_feature.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False),
                             text_feature), dim=1)
        _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False),
                              image_feature), dim=1)

        fusion_image = torch.matmul(_image_h, self.image_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fused_intermedia = fusion_image * fusion_text

        fused_input = torch.matmul(self.fusion_weights,
                                   fused_intermedia.permute(1, 0, 2)).squeeze() + self.fusion_bias
        fused_input = fused_input.view(-1, self.fused_dim)

        return fused_input


if __name__ == '__main__':
    model = LMF()
    model.cuda()

    image_features = torch.rand(8, 512)
    text_features = torch.rand(8, 512)

    image_features = image_features.cuda()
    text_features = text_features.cuda()

    output = model(image_features, text_features)
