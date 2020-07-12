from flask import render_template, flash, redirect, request
from app import app, db
from app.forms import ReviewForm  
from app.models import Review, ReviewAnnotate
import random
from datetime import datetime


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/annotateReview', methods=['GET', 'POST'])
def annotateReview():      
    form = ReviewForm()

    if form.validate_on_submit():
        flash('Thank you! Please select another')
        r = Review.query.get(form.id.data)
        a = ReviewAnnotate(
                label_name=form.topic.data,
                review=r,
                ip=request.remote_addr,
                created_at=str(datetime.now())
            )
        db.session.add(a)
        db.session.commit()

    rowCount = int(Review.query.count())
    review = Review.query.offset(int(rowCount*random.random())).first()

    return render_template('reviews.html', review=review, form=form)