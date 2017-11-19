from pprint import pprint
import random
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd

# Define correlation and post parameters
PERIOD = 86400
OFFSET = 0
MINIMUM_SCORE = 100

# Read bitcoin price data
print('Reading bitcoin price data...')

bitcoin_data = pd.read_csv('btceUSD-no-duplicates.csv', header=0)

# Read all post json files in raw_post_data/
print('Reading post files...')

posts = []
post_file_names = os.listdir('raw_post_data/')
for post_file_name in post_file_names:
    with open('raw_post_data/%s' % post_file_name, 'r') as post_file:
        post = json.loads(post_file.read())
    posts.append(post)

# Define clean and correlation functions
def clean_text(raw_text):
    cleaned_text = raw_text.lower()
    cleaned_text = re.sub(letters_only, ' ', cleaned_text)
    words = [word for word in cleaned_text.split() if word not in english_stop]    
    cleaned_text = ' '.join(words)

    return cleaned_text

def get_price_change(timestamp):
    start = timestamp + OFFSET
    end = timestamp + OFFSET + PERIOD

    start_index = bitcoin_data['time'].searchsorted(start)[0] - 1
    end_index = bitcoin_data['time'].searchsorted(end)[0] - 1

    start_price = bitcoin_data['price'][start_index]
    end_price = bitcoin_data['price'][end_index]

    price_change = end_price - start_price

    return price_change

# Clean and label combined titles and bodies
print('Cleaning and labeling...')

letters_only = re.compile('[^a-z]')
english_stop = stopwords.words('english')

cleaned_posts = []
labels = []

done = 0
total = len(posts)
for post in posts:
    # Check post score
    if post['score'] >= MINIMUM_SCORE:
        raw_text = post['title'] + ' ' + post['selftext']
        cleaned_posts.append(clean_text(raw_text))

        # Label post 1 for price up, 0 for price down
        timestamp = post['timestamp']
        price_change = get_price_change(timestamp)

        if price_change > 0:
            label = 1
        else:
            label = 0

        labels.append(label)

    done += 1
    if done % 100 == 0 or done == total:
        print('\r%s%% Done - %s of %s' % (round((done/total)*100, 1), done, total), end='')

print()

# Vectorize Post Data
print('Vectorizing post data...')

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
vectorized_training_data = vectorizer.fit_transform(cleaned_posts)
vocab = vectorizer.get_feature_names()

# Write data to disk
print('Writing data to disk...')

vectorized_training_data_list = []
for vector in vectorized_training_data:
    vector_list = vector.toarray()[0].tolist()
    vectorized_training_data_list.append(vector_list)

with open('processed_data/cleaned_posts', 'w+') as k:
    k.write(json.dumps(cleaned_posts, indent=4))
with open('processed_data/vectorized_posts', 'w+') as k:
    k.write(json.dumps(vectorized_training_data_list))
with open('processed_data/labels', 'w+') as k:
    k.write(json.dumps(labels))
with open('processed_data/vocab', 'w+') as k:
    k.write(json.dumps(vocab, indent=4))
