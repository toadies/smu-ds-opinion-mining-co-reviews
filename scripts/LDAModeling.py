import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from spacy.lang.en import English
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
import multiprocessing as mp

num_cpus = mp.cpu_count() - 1

parser = English()

with open("../data/tech_review_sent_corpus.pkl","rb") as f:
    tech_review_corpus = pickle.load(f)
    
reviews = pd.DataFrame(tech_review_corpus).review.tolist()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def compute_coherence_values(param):
    lda_model = LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=param["k"], 
                                           random_state=100,
                                           chunksize=100,
                                           workers = num_cpus,
                                           passes=10,
                                           alpha=param["alpha"],
                                           eta=param["beta"],
                                           per_word_topics=True)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=id2word, coherence='c_v')
    
    param["coherence"] = coherence_model_lda.get_coherence()

    return param

if __name__ == "__main__":

    print("Total workers:", num_cpus)

    print("Tokenize the corpus")
    with Pool() as p:
        processed_docs = list(tqdm(p.imap(tokenize, reviews), total=len(reviews)))

    # Create Dictionary
    id2word = corpora.Dictionary(processed_docs)
    # Term Document Frequency
    print("Create a Dictionary")
    corpus = [id2word.doc2bow(text) for i, text in tqdm(enumerate(processed_docs), total=len(processed_docs))]

    grid = {}
    grid['Validation_Set'] = {}
    # Topics range
    min_topics = 3
    max_topics = 15
    step_size = 1
    topics_range = range(max_topics, min_topics, -1)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    parameters = []
    for k in topics_range:
        for a in alpha:
            for b in beta:
                parameters.append({
                        "k":k
                        ,"alpha":a
                        ,"beta":b
                        ,"workers":4
                    })

    print("Running modeling")
    print("Total Paramters", len(parameters))

    results = list(map(compute_coherence_values, tqdm(parameters[:20])))

    df = pd.DataFrame(results).to_csv("../data/lda_modeling.csv",index = False)

