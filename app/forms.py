from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, BooleanField, SubmitField, FileField
from wtforms.validators import DataRequired


class Task1Form(FlaskForm):
    infile = FileField()
    username = StringField('Username', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class Task2Form(FlaskForm):
    infile = FileField()
    submit = SubmitField('Submit')

class Task3Form(FlaskForm):
    infile = FileField()
    username = StringField('Username', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class Task4Form(FlaskForm):
    infile = FileField()
    alpha = IntegerField("Alpha")
    beta = IntegerField("Beta")
    zoom = IntegerField("Zoom")
    submit = SubmitField('Submit')


class Task5Form(FlaskForm):
    infile = FileField()
    isField = BooleanField('Is Field?')
    submit = SubmitField('Submit')


class Task6Form(FlaskForm):
    infile = FileField()
    submit = SubmitField('Sign In')
