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

    def __init__(self, input_dim=512, input_text_dim=768, latent_dim=20, rank=4, drop_prob=0.2,
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
        self.input_text_dim = input_text_dim
        self.fused_dim = encoder_neurons[0]

        self.image_batch_norm = nn.BatchNorm1d(self.input_dim, affine=False, momentum=False,
                                               track_running_stats=True)
        self.text_batch_norm = nn.BatchNorm1d(self.input_text_dim, affine=False, momentum=False,
                                              track_running_stats=True)

        self.text_head = nn.Sequential(nn.Linear(self.input_text_dim, self.input_dim),
                                       nn.ReLU())

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
        self.drop_prob = drop_prob

        self.encoder = nn.Sequential()
        in_features = self.fused_dim

        for (i, neurons) in enumerate(self.encoder_neurons):
            layer = nn.Linear(in_features=in_features, out_features=neurons)
            self.encoder.add_module("encoder%d" % i, layer)
            self.encoder.add_module("encoder_relu%d" % i, nn.ReLU())
            self.encoder.add_module("encoder_dropout%d" % i, nn.Dropout(self.drop_prob))
            in_features = neurons

        # Create mu and sigma of latent variables
        self.z_mean = nn.Linear(in_features, latent_dim)
        self.z_log = nn.Linear(in_features, latent_dim)
        in_features = latent_dim

        self.decoder = nn.Sequential()

        for (i, neurons) in enumerate(self.decoder_neurons):
            layer = nn.Linear(in_features=in_features, out_features=neurons)
            self.decoder.add_module("decoder%d" % i, layer)
            self.decoder.add_module("decoder_relu%d" % i, nn.ReLU())
            self.decoder.add_module("decoder_dropout%d" % i, nn.Dropout(self.drop_prob))
            in_features = neurons

        self.final_layer = nn.Sequential(
            nn.Linear(in_features, self.input_dim + self.input_text_dim),
            # nn.Tanh()
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Generate latent features
        :param mu:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

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

        # use batchnorm to standardize data
        image_feature_norm = self.image_batch_norm(image_feature)
        text_feature_norm = self.text_batch_norm(text_feature)
        text_feature_norm_head = self.text_head(text_feature_norm)

        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False),
                             text_feature_norm_head), dim=1)
        _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False),
                              image_feature_norm), dim=1)

        fusion_image = torch.matmul(_image_h, self.image_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fused_intermedia = fusion_image * fusion_text

        fused_input = torch.matmul(self.fusion_weights,
                                   fused_intermedia.permute(1, 0, 2)).squeeze() + self.fusion_bias
        fused_input = fused_input.view(-1, self.fused_dim)

        # VAE part
        encoder = self.encoder(fused_input)

        z_mean, z_log = self.z_mean(encoder), self.z_log(encoder)
        latent = self.reparameterize(z_mean, z_log)

        decoder = self.decoder(latent)

        output = self.final_layer(decoder)

        return {"reconstruct_input": output, "mean": z_mean, "var": z_log,
                "normalized_image_input": image_feature_norm,
                "normalized_text_input": text_feature_norm}


if __name__ == '__main__':
    model = LMF()
    model.cuda()
    model.train()

    for j in range(100):
        image_features = torch.rand(8, 512) + 0.2
        text_features = torch.rand(8, 768)

        image_features = image_features.cuda()
        text_features = text_features.cuda()

        outputs = model(image_features, text_features)
