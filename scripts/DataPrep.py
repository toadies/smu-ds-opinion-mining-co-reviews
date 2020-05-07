import pandas as pd
import os
import time
from multiprocessing import Pool
import string
import re
from langdetect import detect
import psutil
import pickle
import pathlib as Path
import multiprocessing as mp

#Use all but one CPU
num_cpus = mp.cpu_count() - 1

replacement_words = {
    "it":"technology"
    ,"qa":"quality"
    ,"tech":"technology"
    ,"rep":"representative"
}

stop_words = [
    "the"
    ,"i"
    ,"ii"
    ,"iii"
    ,"junior"
    ,"senior"
    ,"to"
    ,"for"
    ,"the"
    ,"of"
    ,"and"
    ,"in"
    ,"jr"
    ,"sr"
    ,"junior"
    ,"senior"
]

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
def cleanTitle(title):
    title = str(title).translate(translator)
    title = re.sub(' +', ' ',title)
    title = title.lower()
    title_split = title.split(" ")
    for key, value in replacement_words.items():
        title_split = [ value if key == word else word for word in title_split if word not in stop_words and len(word) > 2 ]
    return " ".join(title_split)

def detectLang(str):
    try:    
        return detect(str)
    except:
        return None

if __name__ == "__main__":
    start_time = time.time()

    #Get list of files
    reviews_path = "../data/Database/"
    files = [ reviews_path + i for i in os.listdir(reviews_path) if i.endswith("csv")]
    print("Files to be imported")
    print(list(files))

    #Load in to Pandas Dataframe
    li = []

    for f in files:
        data = pd.read_csv(f)
        data["industry"] = f.split("/")[3][:-4]
        li.append(data)

    reviews = pd.concat(li, axis=0, ignore_index=True)

    print("Total Dataset:",reviews.shape)

    print("Remove Duplicates and NAs")
    sub_time = time.time()
    reviews_ = reviews.copy()
    reviews_ = reviews_.drop(columns=["Unnamed: 0"], axis=1)

    reviews_ = reviews_.drop_duplicates()

    indicesNa = reviews_.loc[reviews_.job_title.isna(),:].index
    reviews_ = reviews_.drop(indicesNa, axis=0)

    #removing none alphanumeric reviews 
    indices = reviews_.loc[reviews_.review.str.isalnum(),:].index
    reviews_ = reviews_.drop(indices, axis=0)
    reviews = reviews_
    reviews_ = None #clear it

    print("    Job Titles NAs",reviews["job_title"].isna().sum())
    print("    Total Records", reviews.shape)
    print("Finished",(time.time() - sub_time))

    print("Clean Job Titles")
    
    sub_time = time.time()
    clean_job_titles = reviews.job_title.tolist()
    # reviews["clean_job_title"] = list( map(cleanTitle, clean_job_titles) )
    with Pool(num_cpus) as p:
        reviews["clean_job_title"] = list( p.map(cleanTitle, clean_job_titles) )

    print(reviews["clean_job_title"].head())
    print("Finished",(time.time() - sub_time))
    
    print("Create a Language Column")
    sub_time = time.time()

    review_content = reviews.review.tolist()
    with Pool(num_cpus) as p:
        reviews["language"] = list( p.map( detectLang, review_content ) )

    LangInxNa = reviews.loc[reviews.language.isna(),:].index
    reviews = reviews.drop(LangInxNa, axis=0)

    print("    Lang NAs",reviews["language"].isna().sum())
    print("    Total Records", reviews.shape)
    print(reviews[["review","language"]].head())
    print("Finished", (time.time() - sub_time))

    with open("../data/all_reviews.pkl","wb") as f:
        pickle.dump(reviews,f)

    print("\nFinished",(time.time() - start_time))