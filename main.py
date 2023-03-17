from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import pycaret.nlp as nlp
import pycaret.classification as classify
import pandas as pd
import numpy as np
from datetime import date
import csv

app = Flask(__name__, template_folder='pages')
# clustering_model = load_model('models/lda_unsupervised')
classification_model = classify.load_model('models/lr_baseline')

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

""" 
Hardcoded the UserName and ScreenName
they are integers generated based on the number of data in the dataset.

"""

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

        combined_df = data.append(user_df, ignore_index=True)

    stopwords=['covid', 'co', 'https', 'pandemic', 'corona', 'virus','coronavirus', 'pandemic', 'go']
    setup_nlp = nlp.setup(data=combined_df, target='OriginalTweet', session_id=123, custom_stopwords=stopwords)
    lda= nlp.create_model('lda', num_topics=4, multi_core=True)
    lda_results = nlp.assign_model(lda)

    lda_results.to_csv('results.csv')

    user_data = lda_results.iloc[-1]

    user_data = user_data.tolist()

    final = pd.DataFrame([user_data], columns=cols_supervised)

    predict = classify.predict_model(classification_model, data=final, round=0)

    label = predict.Label[0]


    return render_template('index.html', pred='This tweet presents a {} Sentiment'.format(label), date=today)