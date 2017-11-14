import pandas as pd
from pprint import pprint
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

PERIOD = 86400 # Period to take change in BTC. 1 day, in seconds
OFFSET = 0 # Offset of ZERO is the perfect lookahead.
MINIMUM_SCORE = 100

# Load Reddit Data CSV
# id,date,text,score,numcomments
reddit_data = pd.read_csv('post_data.csv', header=0)
print('Loaded Reddit Data')

# Define Cleaning Functions
letters_only = re.compile('[^a-xA-Z]')
english_stop = stopwords.words('english')

def clean_text(review):
    cleaned_review = BeautifulSoup(review, 'html5lib').get_text()
    cleaned_review = re.sub(letters_only, ' ', cleaned_review)
    cleaned_review = cleaned_review.lower()
    
    words = [word for word in cleaned_review.split() if word not in english_stop]
    
    cleaned_review = ' '.join(words)
    
    return cleaned_review

# Load Bitcoin Price Data
# timestamp,price,volume
bitcoin_data = pd.read_csv('btceUSD.csv', header=0)
print('Loaded Bitcoin Data')

def get_price_change(timestamp):
    start = timestamp + OFFSET
    end = timestamp + OFFSET + PERIOD

    start_index = bitcoin_data['time'].searchsorted(start)[0]
    end_index = bitcoin_data['time'].searchsorted(end)[0]

    start_price = bitcoin_data['price'][start_index]
    end_price = bitcoin_data['price'][end_index]

    price_change = end_price - start_price

    return price_change

# Clean and Label Posts

training_data_csv = open('training_data.csv', 'w+')
training_data_csv.write('post,label\n')

done = 0
total = len(reddit_data)
for post in reddit_data.values:
    # id,date,text,score,numcomments
    if post[3] >= MINIMUM_SCORE:
        # Get timestamp and clean post
        timestamp = post[1]
        try:
            cleaned_post = clean_text(post[2])
        except:
            print('error: ',type(post[2]), post[2])
            print(done)

        # Ignore empty clean posts
        if not cleaned_post:
            continue

        # 0 for price went down, 1 for price went up
        if get_price_change(timestamp) > 0:
            label = 1
        else:
            label = 0

        training_data_csv.write('%s,%s\n' % (cleaned_post, label))

    done += 1
    if done % 100 == 0:
        print('%s of %s' % (done, total))
        training_data_csv.flush()

training_data_csv.close()
