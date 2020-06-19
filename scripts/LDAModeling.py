import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing as mp
import sys
import os

project_path = os.path.join(os.path.dirname(__file__), "..")
print(project_path)
num_cpus = mp.cpu_count()


def tokenize(doc):
    tokens = doc.split(" ")
    tokens = [word for word in tokens if len(word.strip()) > 0]
    return tokens


if __name__ == "__main__":

    print("Loading tech corpus")
    with open(os.path.join(project_path, "data/tech_review_word_corpus_nn_adj_adv.pkl"), "rb") as f:
        tech_review_corpus = pickle.load(f)
    reviews = pd.DataFrame(tech_review_corpus).review.tolist()

    print("Total workers:", num_cpus)

    print("Tokenize the corpus")

    with open(os.path.join(project_path, "data/stop_words.json"), "r") as f:
        stop_words = json.load(f)

    vectorizer = CountVectorizer(
        min_df=3, max_df=.90, tokenizer=tokenize, stop_words=stop_words, ngram_range=(1, 2))
    X = vectorizer.fit_transform(reviews)
    print("Total Vocab Size", len(vectorizer.vocabulary_))

    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    print(sorted(words_freq, key=lambda x: x[1], reverse=True)[:50])

    topics_range = range(15, 6, -1)
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append(None)
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append(None)

    parameters = []
    for k in topics_range:
        for a in alpha:
            for b in beta:
                parameters.append({
                    "k": k, "alpha": a, "beta": b
                })

    print("Total parameter values to train", len(parameters))

    result = []

    for param in tqdm(parameters):
        aspect_file_name = "results/LDA/lda-aspect-k-{0}-a-{1}-b-{2}-reduced_words.json".format(
            str(param["k"])[:4],
            str(param["alpha"])[:4],
            str(param["beta"])[:4]
        )

        lda = LatentDirichletAllocation(
            learning_method="batch",
            random_state=100,
            n_components=param["k"],
            doc_topic_prior=param["alpha"],
            topic_word_prior=param["beta"],
            n_jobs=num_cpus - 2,
            max_iter=50
        )

        lda.fit(X)

        score = metric_coherence_gensim(measure='u_mass', 
            top_n=25, 
            topic_word_distrib=lda.components_, 
            dtm=X, 
            vocab=np.array([x for x in vectorizer.vocabulary_.keys()]),
            return_mean=True
        )
        
        result.append({
            "k":param["k"],
            "alpha":param["alpha"],
            "beta":param["beta"],
            "score":score
        })

        pd.DataFrame(result).to_csv(os.path.join(project_path,"data/lda_noun_umass.csv"))

    print("Done!")
