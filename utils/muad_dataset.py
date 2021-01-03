# -*- coding: utf-8 -*-
"""
Customized dataset class for muad
"""
import os
import sys

import pandas as pd
import torch.multiprocessing as multiprocessing
from img2vec_pytorch import Img2Vec
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models import common


class MuadDataset(Dataset):
    """Muad Dataset class inherited from Pytorch Dataset for efficient data pipeline
      Parameters
      ----------
      image_path: string
           The path that contains the images
      excel_path : string
           Where is the data excel location, ie: ../data/CrisisMMD_v2.0/train_cleaned.xlsx
           TODO: This is a temporary solution just for CrisisMMD dataset
      image_model_name: string
           Torchvision pretrained model name, only 'resnet-18' and alexnet supported for now
      text_model_name: string
           NLP model to process sentences. Only 'word2vec' and 'bert' supported
      Attributes
      ----------
      image_loc_list : list of string
           Containing all images' paths
      sentences: list of string
           Containing all raw texts
      text_model: model object
           The NLP model we are using. Only Word2Vec and Bert are supported
      image_model: model object
           The image model we are using. Only resnet-18 and alexnet  are supported
      label: list of int
           1 indicate abnormal while 0 represents normal datapoint
      """

    def __init__(self, image_path, excel_path, image_model_name='resnet-18', text_model_name='bert'):
        df = pd.read_excel(excel_path)

        """
        Image Feature Extraction 
        """
        self.image_loc_list = []

        for index, row in df.iterrows():
            image_name = os.path.join(image_path, row["image"] if os.name != 'nt'
                                      else row["image"].replace('/', "\\"))
            self.image_loc_list.append(image_name)

        # TODO: Make cuda parameterized
        self.image_model = Img2Vec(cuda=True, model=image_model_name)

        """
        Text Raw Info
        """
        self.sentences = df["tweet_text"].values
        if text_model_name == 'bert':
            self.text_model = common.Bert()
        elif text_model_name == 'word2vec':
            self.text_model = common.Word2Vec()
        else:
            sys.exit("Invalid NLP model option. Only bert and word2vec is upp")

        assert(len(self.image_loc_list) == len(self.sentences))

        self.label = df["anomaly"].values

    def __len__(self):
        return len(self.image_loc_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Indexes of data

        Returns:
            dictionary: containing the image feature, language feature and label
        """
        # To save memory, we put the image feature extraction part in data retrieval stage
        image_pil = Image.open(self.image_loc_list[idx]).convert('RGB')
        # TODO: The dimension shouldn't be hardcoded in the future
        image_feature = self.image_model.get_vec(image_pil)

        sentence = self.sentences[idx]
        text_feature = self.text_model.extract(sentence)

        label = self.label[idx]

        return {'image_features': image_feature, 'text_features': text_feature, 'label': label}


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    dataset = MuadDataset("../data/CrisisMMD_v2.0/CrisisMMD_v2.0",
                          "../data/CrisisMMD_v2.0/train_cleaned.xlsx")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image_features'].size())
        print(sample_batched['text_features'])
