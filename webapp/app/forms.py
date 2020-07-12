from flask_wtf import FlaskForm
from wtforms import HiddenField, SelectField, SubmitField
from wtforms.validators import DataRequired


class ReviewForm(FlaskForm):
    id = HiddenField()
    topic = SelectField('Select Topic',
                choices=[    
                    ("","Select a Topic"),
                    ("0", "Challenging Work"),
                    ("1", "Job Security & Compensation"),
                    # ("2", "Environment & Culture"),
                    ("3", "Technology"),
                    ("4", "Management"),
                    ("5", "Environment & Culture"),
                    ("6", "Learning & Development"),
                    ("7", "Positive Corporate Outlook")],
                validators=[DataRequired()]
            )
    submit = SubmitField('Submit')