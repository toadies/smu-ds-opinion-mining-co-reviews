import logging
import os
import re
import numpy as np
import gensim
from sklearn.cluster import KMeans
import pymorphy2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class W2VEmbReader:
    def __init__(self, emb_path):
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        self.model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)
        # print(type(self.model))
        self.emb_dim = self.model.vector_size
        self.vocab_size = len(self.model.vocab)

        # emb_matrix = np.zeros((len(reverse_word_map),self.model.vector_size))
        # self.embeddings = {}

        # for word, idx in reverse_word_map.items():
        #     if idx < 3:
        #         pass
        #     else:
        #         # emb_matrix[idx] = self.model[word]
        #         self.embeddings[word] = list(model[word])

        # emb_matrix[2] = self.model['unk']
        # emb_matrix[1] = np.ones(self.emb_dim)

        # self.emb_matrix = emb_matrix
        # self.norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        # self.aspect_size = None
        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))

    # def get_emb_given_word(self, word):
    #     try:
    #         return self.model[word]
    #     except KeyError:
    #         return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.model[word]
                counter += 1
            except KeyError:
                pass

        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_aspect_matrix(self, vocab, n_clusters=0):
        self.aspect_size = n_clusters
        km = KMeans(n_clusters=n_clusters)

        emb_matrix = []
        for word, index in vocab.items():
            try:
                emb_matrix.append(self.model[word])
            except KeyError:
                pass

        km.fit(emb_matrix)
        km_aspects = km.cluster_centers_

        aspects = km_aspects
        # L2 normalization
        norm_aspect_matrix = aspects / np.linalg.norm(aspects, axis=-1, keepdims=True)
        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim
