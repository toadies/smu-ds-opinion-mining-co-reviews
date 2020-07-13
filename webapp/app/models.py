from datetime import datetime
from app import db


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, index=True, unique=True)
    review = db.Column(db.Text)
    lda_topic_label = db.Column(db.Integer)
    lda_topic_name = db.Column(db.String(50))
    keywords = db.Column(db.Text)
    annotates = db.relationship('ReviewAnnotate', backref='review', lazy='dynamic')

    def __repr__(self):
        return '<Index {}>'.format(self.index)


class ReviewAnnotate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    label_name_primary = db.Column(db.Integer)
    label_name_secondary = db.Column(db.Integer)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'))
    created_at = db.Column(db.String(50)) #, default=datetime.utcnow()
    ip = db.Column(db.String(255))


class Sentence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, index=True)
    review = db.Column(db.Text)
    abae_topic_label = db.Column(db.Integer)
    abae_topic_name = db.Column(db.String(50))
    annotates = db.relationship('SentenceAnnotate', backref='sentence', lazy='dynamic')

    def __repr__(self):
        return '<Index {}>'.format(self.index)


class SentenceAnnotate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    label_name_primary = db.Column(db.Integer)
    label_name_secondary = db.Column(db.Integer)
    review_id = db.Column(db.Integer, db.ForeignKey('sentence.id'))
    created_at = db.Column(db.String(50)) #, default=datetime.utcnow()
    ip = db.Column(db.String(255))