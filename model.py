from wtforms import SubmitField, StringField, validators, TextAreaField
from wtforms.widgets import TextArea
from flask_wtf import Form


class RegForm(Form):
    link = StringField('Link',
                        [validators.DataRequired(), validators.Length(min=6, max=256)])
    link_text = TextAreaField('LinkText', render_kw={"rows": 6, "cols": 11})
    result = StringField('Result')
    submit = SubmitField('Submit')