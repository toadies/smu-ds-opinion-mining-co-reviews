import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from string import punctuation
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from gensim.models import Word2Vec
import json
import os
import logging
import re


project_path = os.path.join(os.path.dirname(__file__), "..")

num_cpus = mp.cpu_count() - 1

with open(os.path.join(project_path, "data/all_reviews.pkl"), "rb") as f:
    all_reviews = pickle.load(f)

job_filter = pd.read_csv(os.path.join(project_path, "data/filter_job_titles.csv"))

with open(os.path.join(project_path, "data/replacement_words.json"), "r") as f:
    replacement_words = json.load(f)

lmtzr = WordNetLemmatizer()


def replaceWord(word):
    try:
        word = lmtzr.lemmatize(replacement_words[word])
    except:
        pass
    return word


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
                  alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")

    sentences = [s.strip() for s in sentences]
    sentences = [i for i in sentences if len(i) > 0]
    return sentences



lmtzr = WordNetLemmatizer()
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
punct_replace = re.compile(r'^[-/*+]+|[-/*+]+$') 


def parseSentence(args):
    line = args[0]
    i = args[1]
    result = []
    sent_tokens = split_into_sentences(line.lower())
    for sent in sent_tokens:
        text_token = word_tokenize(sent)
        #Remove punctuation from word, split by /
        text_token = [ punct_replace.subn('', word)[0] for word in text_token]
        text_token = [ word.replace("/", " ") for word in text_token]
        text_token = " ".join(text_token) #Merge back
        
        text_token = text_token.split() #Start Over
        text_lmtz = [lmtzr.lemmatize(token, tag_map[tag[0]])
                        for token, tag in pos_tag(text_token) if (tag not in punctuation) & (not (not token.isalnum()) & (len(token) < 3))]
        text_replacments = [replaceWord(word) for word in text_lmtz]
        if len(text_replacments) > 0:  # Don't append if missing words
            result.append(' '.join(text_replacments).strip())
    
    return i, result


def parseReview(args):
    i, sent_tokens = parseSentence(args)
    return i, " ".join(sent_tokens)


def createStopWords():
    # Create Stop Words
    company_name = all_reviews_en.company_name
    company_name = list(set(company_name))

    tokens = map(str.lower, company_name)
    tokens = map(str.split, tokens)

    stop_words = stopwords.words('english')
    logger.info("Initial Stop Word Count: {0}".format(len(stop_words)))
    for token in tokens:
        stop_words.extend(token[0:1])

    stop_words = list(set(stop_words))
    stop_words = [x for x in stop_words if x not in [
        'management', 'performance', 'measurement']]

    stop_words.extend(
        ["saas", "inc", "company", "chrysler", "packard", "capegemini", "wa", "ha", "im"])
    logger.info("Stop Word Count: {0}".format(len(stop_words)))

    with open(os.path.join(project_path, "data/stop_words.json"), "w") as f:
        json.dump(stop_words, f)

    return stop_words


def filterEmptyString(row):
    tokens = row["review"].split(" ")
    review = " ".join(
        [word for word in tokens if word not in stop_words]).strip()
    return len(review) > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    job_filters = job_filter.clean_job_title.tolist()

    idx = (all_reviews.language == "en")
    all_reviews_en = all_reviews.loc[idx, :]

    idx = (all_reviews_en.clean_job_title.isin(job_filters))
    tech_reviews = all_reviews_en.loc[idx, :].reset_index()

    logger.info("Total Rows in Corpus: {0}".format(str(tech_reviews.shape)))

    print("Create Stop Words")
    stop_words = createStopWords()

    indices = tech_reviews["index"].tolist()
    co_reviews = tech_reviews.review.tolist()
    logger.info("Parse Sentences")
    with Pool(num_cpus) as p:
        tech_review_sent_corpus = list(tqdm(p.imap(parseSentence, zip(co_reviews, indices)), total=len(co_reviews)))

    logger.info("Review Parse")
    with Pool(num_cpus) as p:
        tech_review_word_corpus = list(tqdm(p.imap(parseReview, zip(co_reviews, indices)), total=len(co_reviews)))


    print("Original Review\n", [review for review,
                                i in zip(co_reviews, indices) if i == 119065])
    print("Review Parse\n", [
          review for review in tech_review_word_corpus if review[0] == 119065])
    print("Sentence Parse\n", [
          review for review in tech_review_sent_corpus if review[0] == 119065])

    logger.info("Save Files")

    corpus = [{"index": review[0], "review":review[1]}
              for review in tech_review_word_corpus]

    with open(os.path.join(project_path, "data/tech_review_word_corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    logger.info(
        "Total Records for Review Corpus: {0}".format(str(len(corpus))))

    corpus = [{"index": review[0], "review": sent}
              for review in tech_review_sent_corpus for sent in review[1]]
    corpus = list(filter(filterEmptyString, corpus))
    with open(os.path.join(project_path, "data/tech_review_sent_corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    logger.info(
        "Total Records for Review Corpus: {0}".format(str(len(corpus))))

    logger.info("Create a Pretrained W2V Model")
    logger.info("Parse Sentences")
    all_co_reviews = all_reviews_en.review.tolist()

    with Pool() as p:
        all_review_sent_corpus = list(tqdm(p.imap(parseSentence, zip(all_co_reviews,range(len(all_co_reviews)))), total=len(all_co_reviews)))

    logger.info("Train Model (can take awhile!)")
    sentences = [item.split() for sublist in all_review_sent_corpus for item in sublist[1]]
    model = Word2Vec(sentences, size=200, window=10, min_count=5, workers=num_cpus)
    model.save(os.path.join(project_path, "models/w2v_embedding"))

