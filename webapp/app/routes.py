from flask import render_template, flash, redirect, request
from app import app, db
from app.forms import ReviewForm, SentenceForm
from app.models import Review, ReviewAnnotate, Sentence, SentenceAnnotate
import random
from datetime import datetime


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/annotateReview', methods=['GET','POST'])
def annotateReview():      
    form = ReviewForm()

    if form.validate_on_submit():
        choices = form.topic_primary.choices
        primary = [ topic[1] for topic in choices if form.topic_primary.data == topic[0]][0]
        secondary = [ topic[1] for topic in choices if form.topic_secondary.data == topic[0]][0]

        flash('You have selected {0} and {1} for {2}.'.format(primary, secondary, form.id.data))

        r = Review.query.get(form.id.data)
        a = ReviewAnnotate(
                label_name_primary=form.topic_primary.data,
                label_name_secondary=form.topic_secondary.data,
                review=r,
                ip=request.remote_addr,
                created_at=str(datetime.now())
            )
        db.session.add(a)
        db.session.commit()  
        return redirect('annotateReview')

    rowCount = int(Review.query.count())
    review = Review.query.offset(int(rowCount*random.random())).first()

    return render_template('reviews.html', review=review, form=form)


@app.route('/annotateSentence', methods=['GET', 'POST'])
def annotateSentence():      
    form = SentenceForm()

    if form.validate_on_submit():
        choices = form.topic_primary.choices
        primary = [ topic[1] for topic in choices if form.topic_primary.data == topic[0]][0]
        secondary = [ topic[1] for topic in choices if form.topic_secondary.data == topic[0]][0]

        flash('You have selected {0} and {1} for {2}.'.format(primary, secondary, form.id.data))

        s = Sentence.query.get(form.id.data)
        a = SentenceAnnotate(
                label_name_primary=form.topic_primary.data,
                label_name_secondary=form.topic_secondary.data,
                sentence=s,
                ip=request.remote_addr,
                created_at=str(datetime.now())
            )
        db.session.add(a)
        db.session.commit()
        return redirect('annotateSentence')

    rowCount = int(Sentence.query.count())
    sentence = Sentence.query.offset(int(rowCount*random.random())).first()

    return render_template('sentence.html', sentence=sentence, form=form)