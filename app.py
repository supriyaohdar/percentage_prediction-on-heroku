import flask
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    int_feature = [int(x) for x in request.form.values()]
    prediction = model.predict([int_feature])

    return render_template('index.html', prediction_text='The percentage of marks is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)