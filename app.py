from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

import json
import os
import nltk
import random

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


# english_bot.set_trainer(ChatterBotCorpusTrainer)
# english_bot.train("chatterbot.corpus.english")
def predict_sentence(sent):
    word_count = len(nltk.word_tokenize(sent))
    scores = get_score()
    labels = get_labels(scores)
    response = {
        'sentiment': {
            "anticipation": scores[0],
            "joy": scores[1],
            "trust": scores[2],
            "fear": scores[3],
            "surprise": scores[4],
            "sadness": scores[5],
            "disgust": scores[6],
            "anger": scores[7],
        },
        'label': labels,
        "wordCount": word_count,
        "sent": sent
    }
    return response

def get_score():
    l = []
    for i in range(8):
        l.append(random.random())
    return l

def get_labels(scores):
    labels = ["anticipation", "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger"];
    answers = []
    for i, s in enumerate(scores):
        if i > 0.7:
            answers.append(labels[i])
    return answers



def predict_sentiment(text, type):
    if type == "document":
        pred = predict_sentence(text)
        return json.dumps([pred])
    else:
        sentences = nltk.sent_tokenize(text)
        response = []
        response.append(predict_sentence(text))
        for sent in sentences:
            response.append(predict_sentence(sent))
        return json.dumps(response)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('messageText')
    type = request.args.get('type')
    pred = predict_sentiment(userText, type)

    return pred



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['photo']
        # file_name = photos.save(file)
        return json.dumps({'answer': "%s has received!"%file.name})



if __name__ == "__main__":
    app.run()
