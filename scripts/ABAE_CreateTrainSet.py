import json
import os, sys

project_path = os.path.join(os.path.dirname(__file__), "..")
if project_path not in sys.path:
    sys.path.append(project_path+"/scripts")
    sys.path.append(project_path+"/classes")

import pickle
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import ABAE.reader as dataset
import numpy as np
import json

def removeStopWords(review):
    tokens = review.split(" ")
    return " ".join([ word for word in tokens if word not in stop_words ])


if __name__ == "__main__":
    with open(os.path.join(project_path, "data/tech_review_sent_corpus.pkl"),"rb") as f:
        tech_review_corpus = pickle.load(f)

    print("Initial Corpus Size", len(tech_review_corpus))

    with open(os.path.join(project_path, "data/stop_words.json"), "r") as f:
        stop_words = json.load(f)

    reviews = [review["review"] for review in tech_review_corpus]

    with Pool() as p:
        reviews = list(tqdm(p.imap(removeStopWords, reviews), total=len(reviews)))

    word_length = [len(review) for review in reviews]

    print("Max Character length", max(word_length))
    print("Average Character Length", np.mean(word_length))
    print("Standard Deviation", np.std(word_length))
    print("Median Character Length", np.median(word_length))
    print("Words 2 standard deviations from mean", np.mean(word_length) + (2*np.std(word_length)))
    
    maxlen = int(round(np.mean(word_length) + (2*np.std(word_length)))) #Based on 2 standard deviations from mean

    vocab, train_x, overall_maxlen = dataset.get_data(reviews, maxlen=maxlen, word_freq=5)

    # train_x = train_x[0:30000]
    print('Number of training examples: ', len(train_x))
    print('Length of vocab: ', len(vocab))

    with open(os.path.join(project_path, "data/abae_train.pkl"),"wb") as f:
        pickle.dump((vocab, train_x, overall_maxlen), f)