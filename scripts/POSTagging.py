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
        print("Split Words")
        reviews_pos = list(tqdm(p.imap(str.split, reviews), total=len(reviews)))
        print("Tagger")
        reviews_pos = list(tqdm(p.imap(pos_tag, reviews_pos), total=len(reviews_pos)))


    vocab = {}

    for review in reviews_pos:
        for word in review:
            if not word[0] in vocab.keys():
                vocab[word[0]] = {}

            try:
                vocab[word[0]][word[1]] += 1
            except:
                vocab[word[0]][word[1]] = 1

    df = pd.DataFrame(vocab).T
    df.fillna(0, inplace=True)
    df["tag"] = df.idxmax(axis=1)
    df = df.reset_index().rename(columns={"index":"word"})
    df.to_csv(os.path.join(project_path,"data/pos-tagger.csv"),index=False)

    print("Done!")
    print(df[["word","tag"]].head())