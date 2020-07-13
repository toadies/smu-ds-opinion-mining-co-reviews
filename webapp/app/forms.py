from flask_wtf import FlaskForm
from wtforms import HiddenField, SelectField, SubmitField
from wtforms.validators import DataRequired


class ReviewForm(FlaskForm):
    id = HiddenField()
    topic_primary = SelectField('Select Primary Topic',
                choices=[    
                    ("","Select a Topic"),
                    ("0", "Collaborative Environment"),
                    ("1", "Job Security & Compensation"),
                    # ("2", "Growth Opportunities"),
                    ("3", "Technical Skills"),
                    ("4", "Hiring Process"),
                    ("5", "Environment & Culture"),
                    ("6", "Learning & Development"),
                    ("7", "Growth Opportunities")],
                validators=[DataRequired()],
                default=""
            )
    topic_secondary = SelectField('Select Secondary Topic',
                choices=[    
                    ("","Select a Topic"),
                    ("0", "Collaborative Environment"),
                    ("1", "Job Security & Compensation"),
                    # ("2", "Growth Opportunities"),
                    ("3", "Technical Skills"),
                    ("4", "Hiring Process"),
                    ("5", "Environment & Culture"),
                    ("6", "Learning & Development"),
                    ("7", "Growth Opportunities")],
                validators=[DataRequired()]
            )
    submit = SubmitField('Submit')


class SentenceForm(FlaskForm):
    id = HiddenField()
    topic_primary = SelectField('Select Primary Topic',
                choices=[    
                    ("","Select a Topic"),
                    ("0", "Support"),
                    ("1", "Management"),
                    ("2", "Technical Skills"),
                    ("3", "Work Life"),
                    ("4", "Human Resource & Benefits")],
                validators=[DataRequired()]
            )
    topic_secondary = SelectField('Select Secondary Topic',
                choices=[    
                    ("","Select a Topic"),
                    ("0", "Support"),
                    ("1", "Management"),
                    ("2", "Technical Skills"),
                    ("3", "Work Life"),
                    ("4", "Human Resource & Benefits")],
                validators=[DataRequired()]
            )
    submit = SubmitField('Submit')