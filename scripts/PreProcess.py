import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from ReplacementWords import replacement_words
from gensim.models import Word2Vec
import json
import os
import logging


project_path = os.path.join(os.path.dirname(__file__),"..")

num_cpus = mp.cpu_count() - 1

with open(os.path.join(project_path,"data/all_reviews.pkl"),"rb") as f:
    all_reviews = pickle.load(f)
    
job_filter = pd.read_csv(os.path.join(project_path,"data/filter_job_titles.csv"))

replacementWords = replacement_words

def replaceWord(word):
    try:
        word = replacementWords[word]
    except:
        pass
    return word

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("\r"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    
    sentences = [s.strip() for s in sentences]
    sentences = [i for i in sentences if len(i) > 0]
    return sentences

lmtzr = WordNetLemmatizer()
def parseSentence(args):
    line = args[0]
    i = args[1]
    result=[]
    sent_tokens = split_into_sentences(line.lower())
    for sent in sent_tokens:
        text_token = re.split(r"\W+",sent)
        # text_rmstop = [lmtzr.lemmatize(word) for word in text_token if word not in stop_words]
        text_replacments = [ replaceWord(word) for word in text_token]
        if len(text_replacments) > 0: # Don't append if missing words
            result.append(' '.join(text_replacments).strip())
    return i, result

def parseReview(args):
    line = args[0]
    i = args[1]
    lmtzr = WordNetLemmatizer()
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_stem = [lmtzr.lemmatize(w) for w in text_token]
    text_replacments = [replaceWord(word) for word in text_stem]
    # text_rmstop = [i for i in text_replacments if i not in stop_words]

    return i, " ".join(text_replacments)

def getStopWords():
    #Create Stop Words
    company_name = all_reviews_en.company_name
    company_name = list(set(company_name))

    tokens = map(str.lower, company_name)
    tokens = map(str.split, tokens)

    stop_words = stopwords.words('english')

    stop_words = []
    for token in tokens:
        stop_words.extend(token[0:1])
        
    stop_words = list( set(stop_words) )
    stop_words = [x for x in stop_words if x not in ['management','performance','measurement']]

    stop_words.extend(["saas","inc","company","chrysler","packard","capegemini"])
    print("Stop word count", len(stop_words))
    return stop_words

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    job_filters = job_filter.clean_job_title.tolist()

    idx = (all_reviews.language == "en")
    all_reviews_en = all_reviews.loc[idx,:]

    idx = (all_reviews_en.clean_job_title.isin(job_filters))
    tech_reviews = all_reviews_en.loc[idx,:].reset_index()

    logger.info("Total Rows in Corpus: {0}".format(str(tech_reviews.shape)))

    # print("Get Stop Words")
    # stop_words = getStopWords()

    indices = tech_reviews["index"].tolist()
    co_reviews = tech_reviews.review.tolist()
    logger.info("Parse Sentences")
    with Pool(num_cpus) as p:
        tech_review_sent_corpus = list(tqdm(p.imap(parseSentence, zip(co_reviews,indices)), total=len(co_reviews)))

    logger.info("Review Parse")
    with Pool(num_cpus) as p:
        tech_review_word_corpus = list(tqdm(p.imap(parseReview, zip(co_reviews,indices)), total=len(co_reviews)))

    print("Original Review\n",[review for review, i in zip(co_reviews, indices) if i == 119065])
    print("Review Parse\n",[review for review in tech_review_word_corpus if review[0] == 119065])
    print("Sentence Parse\n",[review for review in tech_review_sent_corpus if review[0] == 119065])

    logger.info("Save Files")

    corpus = [ {"index":review[0],"review":review[0]} for review in tech_review_word_corpus ]

    with open(os.path.join(project_path,"data/tech_review_word_corpus.pkl"),"wb") as f:
        pickle.dump(corpus, f)

    logger.info("Total Records for Review Corpus: {0}".format(str(len(corpus))))

    corpus = [ { "index":review[0] ,"review":sent } for review in tech_review_sent_corpus for sent in review[1] ]

    with open(os.path.join(project_path,"data/tech_review_sent_corpus.pkl"),"wb") as f:
        pickle.dump(corpus, f)

    logger.info("Total Records for Review Corpus: {0}".format(str(len(corpus))))

    logger.info("Create a Pretrained W2V Model")
    logger.info("Parse Sentences")
    all_co_reviews = all_reviews_en.review.tolist()

    with Pool() as p:
        all_review_sent_corpus = list(tqdm(p.imap(parseSentence, zip(all_co_reviews,range(len(all_co_reviews)))), total=len(all_co_reviews)))

    logger.info("Train Model (can take awhile!)")
    sentences = [item.split() for sublist in all_review_sent_corpus for item in sublist[1]]
    model = Word2Vec(sentences, size=200, window=10, min_count=5, workers=num_cpus)
    model.save(os.path.join(project_path,"models/w2v_embedding"))
