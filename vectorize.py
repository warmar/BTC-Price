import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# post, label
training_data = pd.read_csv('training_data.csv', header=0)

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

post_list = training_data['post'].tolist()

vectorizer.fit_transform(post_list)
vocab = vectorizer.get_feature_names()

print(vocab)
