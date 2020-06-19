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
from nltk import pos_tag
import ABAE.reader as dataset


def filterWords(sent):
    tags = ["N","J","R"]
    tokens = sent.split()
    pos_tokens = pos_tag(tokens)
    tokens = [ word[0] for word in pos_tokens if word[1][0] in tags ]
    return " ".join(tokens)


with open(os.path.join(project_path,"data/stop_words.json"), "r") as f:
  stop_words = json.load(f)


def removeStopWords(review):
  tokens = review.split(" ")
  return " ".join([ word for word in tokens if word not in stop_words ])

if __name__ == "__main__":
    review_corpus_path = os.path.join(project_path, "data/tech_review_sent_corpus.pkl")

    with open(review_corpus_path, "rb") as f:
        tech_review_corpus = pickle.load(f)

    indices = [review["index"] for review in tech_review_corpus]
    reviews = [review["review"] for review in tech_review_corpus]

    print("Parse Reviews")
    with Pool() as p:
        reviews = list(tqdm(p.imap(filterWords, reviews), total=len(reviews)))
        reviews = list(tqdm(p.imap(removeStopWords, reviews), total=len(reviews)))

    tech_review_sent_corpus_nn_adj_adv = [ {"index":idx, "review":sent}
                                 for idx, sent in zip(indices, reviews) if len(sent.split()) > 1]

    print("Final Size", len(tech_review_sent_corpus_nn_adj_adv))

    with open(os.path.join(project_path, "data/tech_review_sent_corpus_nn_adj_adv.pkl"), "wb") as f:
        pickle.dump(tech_review_sent_corpus_nn_adj_adv, f)

    tech_review_word_corpus = {review["index"]: [] for review in tech_review_sent_corpus_nn_adj_adv}
    for review in tech_review_sent_corpus_nn_adj_adv:
        tech_review_word_corpus[review["index"]].append(review["review"])

    tech_review_word_corpus_nn_adj_adv = []
    for key, value in tech_review_word_corpus.items():
        tech_review_word_corpus_nn_adj_adv.append({
            "index":key,
            "review":" ".join(value)
        })

    print("Total Size of Reviews", len(tech_review_word_corpus_nn_adj_adv))
    with open(os.path.join(project_path, "data/tech_review_word_corpus_nn_adj_adv.pkl"), "wb") as f:
        pickle.dump(tech_review_word_corpus_nn_adj_adv, f)


