import pickle
import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

project_path = os.path.join(os.path.dirname(__file__), "..")
if project_path not in sys.path:
    sys.path.append(project_path+"/scripts")
    sys.path.append(project_path+"/classes")
    sys.path.append(project_path+"/webapp")

from app import db
from app.models import Review

if __name__ == "__main__":
    
    with open(os.path.join(project_path,"data/final_review_topics.pkl"), "rb") as f:
        final_review_topics = pickle.load(f)
    print(final_review_topics.columns)
    
    X_train, X_test = train_test_split(final_review_topics, test_size=0.01, random_state=100)

    for i, row in X_test.iterrows():
        r = Review(
            index=int(row["index"]),
            review=row["review"],
            keywords=row["Keywords"],
            lda_topic_label=int(row["lda_topic_label"]),
            lda_topic_name=row["lda_topic_name"]
        )
        db.session.add(r)

    db.session.commit()
