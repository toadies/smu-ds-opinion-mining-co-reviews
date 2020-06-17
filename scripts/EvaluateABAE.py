import json
import os, sys

project_path = os.path.join(os.path.dirname(__file__), "..")
if project_path not in sys.path:
    sys.path.append(project_path+"/scripts")
    sys.path.append(project_path+"/classes")

from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import ABAE.utils as U
import ABAE.reader as dataset
from ABAE.my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import load_model

with open(os.path.join(project_path, "data/tech_review_sent_corpus.pkl"),"rb") as f:
    tech_review_corpus = pickle.load(f)


with open(os.path.join(project_path, "data/stop_words.json"), "r") as f:
    stop_words = json.load(f)


def removeStopWords(review):
    tokens = review.split(" ")
    return " ".join([ word for word in tokens if word not in stop_words])


if __name__ == "__main__":

    out_dir = os.path.join(project_path,"results/ABAE")
    model_name = "abae-k-10-orth-0.3"

    reviews = pd.DataFrame(tech_review_corpus).review.tolist()

    with Pool() as p:
        reviews = list(tqdm(p.imap(removeStopWords, reviews), total=len(reviews)))

    maxlen = 115  # Based on 2 standard deviations from mean
    vocab_path = os.path.join(project_path,"data/vocab-text-review.txt")
    vocab, test_x, overall_maxlen = dataset.get_data(reviews, vocab_path, vocab_size=0, maxlen=maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w

    test_length = test_x.shape[0]
    batch_size = 50
    splits = []
    for i in range(1, test_length // batch_size):
        splits.append(batch_size * i)
    if test_length % batch_size:
        splits += [(test_length // batch_size) * batch_size]
    test_x = np.split(test_x, splits)

    model = load_model(out_dir + '/' + model_name,
                    custom_objects={"Attention": Attention, "Average": Average, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss},
                    compile=True)

    test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
    att_weights, aspect_probs = [], []
    for batch in tqdm(test_x):
        cur_att_weights, cur_aspect_probs = test_fn([batch, 0])
        att_weights.append(cur_att_weights)
        aspect_probs.append(cur_aspect_probs)

    att_weights = np.concatenate(att_weights)
    aspect_probs = np.concatenate(aspect_probs)

    ######### Topic weight ###################################
    topic_weight_out = open(out_dir + '/'+model_name+'-topic_weights', 'wt', encoding='utf-8')
    labels_out = open(out_dir + '/'+model_name+'-labels.txt', 'wt', encoding='utf-8')
    print('Saving topic weights on test sentences...')
    for probs in aspect_probs:
        labels_out.write(str(np.argmax(probs)) + "\n")
        weights_for_sentence = ""
        for p in probs:
            weights_for_sentence += str(p) + "\t"
        weights_for_sentence.strip()
        topic_weight_out.write(weights_for_sentence + "\n")
    print(aspect_probs)

    ## Save attention weights on test sentences into a file
    att_out = open(out_dir + '/'+model_name+'-att_weights', 'wt', encoding='utf-8')
    print('Saving attention weights on test sentences...')
    test_x = np.concatenate(test_x)
    for c in range(len(test_x)):
        att_out.write('----------------------------------------\n')
        att_out.write(str(c) + '\n')

        word_inds = [i for i in test_x[c] if i != 0]
        line_len = len(word_inds)
        weights = att_weights[c]
        weights = weights[(overall_maxlen - line_len):]

        words = [vocab_inv[i] for i in word_inds]
        att_out.write(' '.join(words) + '\n')
        for j in range(len(words)):
            att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')