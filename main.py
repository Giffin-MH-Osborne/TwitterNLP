from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np
from datetime import date
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from afinn import Afinn
from pycaret.classification import *
nltk.download('vader_lexicon')

def sentiment_encoder(sentiment:str):
    if sentiment == 'Extremely Negative':
        return -2
    elif sentiment == 'Negative':
        return -1
    elif sentiment == 'Neutral':
        return 0
    elif sentiment == 'Positive':
        return 1
    elif sentiment == 'Extremely Positive':
        return 2

data = pd.read_csv('data/Corona_NLP_train.csv', encoding='latin-1')

app = Flask(__name__, template_folder='pages')

classification_model = load_model('models/gbc_classifier')

data = pd.read_csv('data/Corona_NLP_train.csv', encoding='latin1')
username = int(data.iloc[-2]['UserName']) + 1
screenname = int(data.iloc[-2]['ScreenName']) + 1

today = date.today()

cols_unsupervised = []

with open('cols_unsupervised.csv', 'r') as f:
    lines = f.read().replace('\ufeff','')
    cols_unsupervised = lines.split(',')

cols_supervised = []

with open('cols_supervised.csv', 'r') as f:
    lines = f.read().replace('\ufeff','')
    cols_supervised = lines.split(',')

@app.route('/')
def home():
    return render_template('index.html', pred='', date=today)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global data
    int_features = [x for x in request.form.values()]

    if(int_features):
        user_data = []

        for i in range(len(int_features)):
            if i == 0:
                user_data.append(str(username))
            elif i == 1:
                user_data.append(str(screenname))
            else:
                user_data.append(str(int_features[i]).replace('\r','').replace('\n','').strip())
        user_data.append('Neutral')
        
        user_df = pd.DataFrame([user_data], columns=cols_unsupervised)

        user_df['encoded_sentiment'] = user_df.apply(lambda row: sentiment_encoder(row['Sentiment']),axis=1)

        sia = SentimentIntensityAnalyzer()

        af = Afinn()

        user_df['neg'] = user_df.apply(lambda row: sia.polarity_scores(row['OriginalTweet'])['neg'],axis=1)
        user_df['neu'] = user_df.apply(lambda row: sia.polarity_scores(row['OriginalTweet'])['neu'],axis=1)
        user_df['pos'] = user_df.apply(lambda row: sia.polarity_scores(row['OriginalTweet'])['pos'],axis=1)
        user_df['compound'] = user_df.apply(lambda row: sia.polarity_scores(row['OriginalTweet'])['compound'],axis=1)
        user_df['affin'] = user_df.apply(lambda row: af.score(row['OriginalTweet']),axis=1)


        predict = predict_model(classification_model, data=user_df, round=0)

        label = predict.Label[0]

        if(label == 0):
            result = 'a Neutral'
        if(label == 1):
            result = 'a Positive'
        if(label == 2):
            result = 'an Extremely Positive'
        if(label == -1):
            result = 'a Negative'
        if(label == -2):
            result = 'an Extremely Negative'


    return render_template('index.html', pred='This tweet presents {} Sentiment'.format(result), date=today)