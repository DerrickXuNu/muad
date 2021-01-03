# -*- coding: utf-8 -*-
"""
Image and language data unimodal feature representation
"""
import gensim.downloader as api
import nltk
import numpy as np
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel


class Word2Vec:
    """Word2Vec model
      Parameters
      ----------
      Attributes
      ----------
      w2v : Model object
          Word2Vec model
      stop_words: set
           Containing all stopping words in english
      """

    def __init__(self):
        nltk.download('stopwords')
        nltk.download("punkt")

        self.w2v = api.load('word2vec-google-news-300')
        self.stop_words = set(stopwords.words('english'))

    def extract(self, sentence):
        """
        Extract feature vector from given sentence
        :param sentence: a string
                         The sentence that needs to be converted
        :return: vec: a numpy array
                      The converted feature vector
        """
        token = word_tokenize(sentence.lower())
        filtered_sentence = [w for w in token if w not in self.stop_words]
        new_words = [word for word in filtered_sentence if word.isalnum()]

        vec_list = []
        for item in new_words:
            try:
                tmp = self.w2v[item]
                vec_list.append(tmp)
            except KeyError:
                continue
        vec = np.array(vec_list).mean(axis=0)

        return vec


class Bert:
    """Word2Vec model
      Parameters
      ----------
      Attributes
      ----------
      tokenizer : BertTokenizer object
          Tokenizer for Bert
      model: BERT model
          Pretrained BERT Model
      """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_embedding(self, sequence):
        """
        Feature embeddings using given model
        @TODO: Make the max_length parameterized
        :param sequence:
        :return: numpy array
        """
        input_ids = torch.tensor(self.tokenizer.encode(sequence,
                                                       add_special_tokens=True,
                                                       max_length=512,
                                                       truncation=True)).unsqueeze(0)
        return self.model(input_ids)[1].data.numpy()

    def extract(self, sentence):
        """
        Extract feature vector from sentence using BERT model
        :param sentence: string
                         The sentence that needs to be converted
        :return: numpy array
        """
        bert_cls_embeding = self.get_embedding(sentence).squeeze()

        return bert_cls_embeding
