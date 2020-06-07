import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os

from nltk import pos_tag

project_path = os.path.join(os.path.dirname(__file__),"..")

if __name__ == "__main__":
    print("Loading tech corpus")
    with open(os.path.join(project_path,"data/tech_review_sent_corpus.pkl"),"rb") as f:
        tech_review_corpus = pickle.load(f)
    reviews = pd.DataFrame(tech_review_corpus).review.tolist()

    with Pool() as p:
        tech_review_word_corpus = list(tqdm(p.imap(pos_tag, reviews), total=len(reviews)))
