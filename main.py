from flask import Flask, request, render_template
from model import RegForm
from ML.TestNewLink import LinkTester


from flask_bootstrap import Bootstrap
import os
app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
@app.route('/result', methods=['GET', 'POST'])
def registration():
    form = RegForm(request.form)
    if request.method == 'POST' and form.validate_on_submit():
        url = form.data["link"]
        link_tester = LinkTester(url)
        is_fake, parsed_text = link_tester.test()
        if is_fake:
            form.result.label.text = "This link is Fake News"
        else:
            form.result.label.text = "This link is Satire"
        form.link_text.data = parsed_text
        return render_template('result.html', form=form)
    return render_template('index_custom.html', form=form)


if __name__ == '__main__':
    app.run()
