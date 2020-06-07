import logging
import os
import sys

src_path = os.path.join(os.path.dirname(__file__))
sys.path.append(src_path)

from numpy.random import seed
import tensorflow

seed(76244)
tensorflow.random.set_seed(76244)

import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm

from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
from w2vEmbReader import W2VEmbReader as EmbReader

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_model(overall_maxlen, vocab, aspect_size, neg_size, emb_reader, ortho_reg_default):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = K.l2_normalize(weight_matrix, axis=-1)
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0])))
        return ortho_reg_default * reg
    
    # ##### Inputs #####
    sentence_input = Input(shape=(overall_maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(neg_size, overall_maxlen), dtype='int32', name='neg_input')

    aspect_matrix = emb_reader.get_aspect_matrix(aspect_size)
    aspect_size = emb_reader.aspect_size
    emb_dim = emb_reader.emb_dim
    
    # ##### Construct word embedding layer #####
    vocab_size = len(vocab)
    word_emb = Embedding(vocab_size, emb_dim,
                         mask_zero=True, name='word_emb',
                         embeddings_constraint=MaxNorm(10))

    ##### Compute sentence representation #####
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    att_weights = Attention(name='att_weights',
                            W_constraint=MaxNorm(10),
                            b_constraint=MaxNorm(10))([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights])

    ##### Compute representations of negative instances #####
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction #####
    p_t = Dense(aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(aspect_size, emb_dim, name='aspect_emb',
                            W_constraint=MaxNorm(10),
                            W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(inputs=[sentence_input, neg_input], outputs=[loss])

    ### Word embedding and aspect embedding initialization ######
    print('Initializing word embedding matrix')
    embs = model.get_layer('word_emb').embeddings
    K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
    print('Initializing aspect embedding matrix as centroid of kmean clusters')
    K.set_value(model.get_layer('aspect_emb').W, aspect_matrix)

    return model
