import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from ReplacementWords import replacement_words

num_cpus = mp.cpu_count() - 1

with open("../data/all_reviews.pkl","rb") as f:
    all_reviews = pickle.load(f)
    
job_filter = pd.read_csv("../data/filter_job_titles.csv")

job_filters = job_filter.clean_job_title.tolist()

def replaceWord(word):
    try:
        return replacement_words[word]
    except:
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
def parseSentence(line):
    result=[]
    sent_tokens = split_into_sentences(line.lower())
    for sent in sent_tokens:
        text_token = re.split(r"\W+",sent)
        text_replacments = [replaceWord(word) for word in text_token]
        text_rmstop = [lmtzr.lemmatize(word) for word in text_replacments if word.lower() not in stop_words]
        result.append(' '.join(text_rmstop).strip())
    return result

def parseReview(line):
    lmtzr = WordNetLemmatizer()
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_replacments = [replaceWord(word) for word in text_token]
    text_rmstop = [i for i in text_replacments if i not in stop_words]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return " ".join(text_stem)

def getStopWords():
    #Create Stop Words
    company_name = all_reviews_en.company_name
    company_name = list(set(company_name))

    tokens = map(str.lower, company_name)
    tokens = map(str.split, tokens)

    stop_words = stopwords.words('english')


    for token in tokens:
        stop_words.extend(token[0:1])
        
    stop_words = list( set(stop_words) )
    stop_words = [x for x in stop_words if x not in ['management','performance','measurement']]

    stop_words.extend(["saas","inc","company","chrysler","packard","capegemini"])
    print("Stop word count", len(stop_words))
    return stop_words

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

if __name__ == "__main__":

    print("Loading Data")
    with open("../data/all_reviews.pkl","rb") as f:
        all_reviews = pickle.load(f)
        
    job_filter = pd.read_csv("../data/filter_job_titles.csv")

    job_filters = job_filter.clean_job_title.tolist()

    idx = (all_reviews.language == "en")
    all_reviews_en = all_reviews.loc[idx,:]

    idx = (all_reviews_en.clean_job_title.isin(job_filters))
    tech_reviews = all_reviews_en.loc[idx,:].reset_index()

    print("Total Rows in Corpus", tech_reviews.shape)

    print("Get Stop Words")
    stop_words = getStopWords()

    co_reviews = tech_reviews.review.tolist()
    print("Parse Sentences")
    with Pool(num_cpus) as p:
        tech_review_sent_corpus = list(tqdm(p.imap(parseSentence, co_reviews), total=len(co_reviews)))

    print("Review Parse")
    with Pool(num_cpus) as p:
        tech_review_word_corpus = list(tqdm(p.imap(parseReview, co_reviews), total=len(co_reviews)))

    print("Original Review\n",co_reviews[0])
    print("Review Parse\n",tech_review_word_corpus[0])
    print("Sentence Parse\n",tech_review_sent_corpus[0])

    print("Save Files")
    indices = tech_reviews["index"].tolist()
    corpus = []
    for idx, review in zip(indices, tech_review_word_corpus):
        corpus.append({
            "index":idx
            ,"review":review
        })

    with open("../data/tech_review_word_corpus.pkl","wb") as f:
        pickle.dump(corpus, f)

    print("Total Records for Review Corpus:", len(corpus))

    corpus = []
    for idx, review in zip(indices, tech_review_sent_corpus):
        for i in range(len(review)):
            corpus.append({
                "index":idx
                ,"review":review[i]
            })

    with open("../data/tech_review_sent_corpus.pkl","wb") as f:
        pickle.dump(corpus, f)

    print("Total Records for Review Corpus:", len(corpus))

    print("Create a Pretrained W2V Model")
    print("Parse Sentences")
    all_co_reviews = all_reviews_en.review.tolist()

    with Pool() as p:
        all_review_sent_corpus = list(tqdm(p.imap(parseSentence, all_co_reviews), total=len(all_co_reviews)))

    print("Train Model (can take awhile!)")
    sentences = [item.split() for sublist in all_review_sent_corpus for item in sublist]
    model = Word2Vec(sentences, size=200, window=10, min_count=5, workers=num_cpus)
    model.save("../models/w2v_embedding")
