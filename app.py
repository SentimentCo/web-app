from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

import json
import os
import nltk

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


# english_bot.set_trainer(ChatterBotCorpusTrainer)
# english_bot.train("chatterbot.corpus.english")

def predict_sentiment(text, type):
    def predict_sentence(sent):
        word_count = len(nltk.word_tokenize(sent))
        response = {
        "sentiment": {
            "label": ["anticipation", "Joy"],
            "anticipation": 0.941,
            "joy": 0.73,
            "trust": 0.59,
            "fear": 0.01,
            "surprise": 0.042,
            "sadness": 0.05,
            "disgust": 0.03,
            "anger": 0.01,
        },
        "wordCount": word_count
        }
        return response

    if type == "document":
        pred = predict_sentence(text)
        return json.dumps([pred])

    else:
        sentences = nltk.sent_tokenize(text)
        response = []
        for sent in sentences:
            response.append(predict_sentence(sent))
        return json.dumps(response)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('messageText')
    print(request.args)
    type = request.args.get('type')

    pred = predict_sentiment(userText, type)
    # score = 0.0
    # topic = 'shopping'

    return json.dumps(pred)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['photo']
        # file_name = photos.save(file)
        return json.dumps({'answer': "%s has received!"%file.name})



if __name__ == "__main__":
    app.run()
