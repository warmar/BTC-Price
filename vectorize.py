import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# post, label
training_data = pd.read_csv('raw_training_data.csv', header=0)

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

# Load posts and labels as lists
post_list = training_data['post'].tolist()
label_list = training_data['label'].tolist()

# Remove any broken data points
for i, post in reversed(list(enumerate(post_list))):
    if not type(post)==str:
        del post_list[i]
        del label_list[i]

# Vectorize posts
vectorized_training_data = vectorizer.fit_transform(post_list)
vocab = vectorizer.get_feature_names()

training_data_csv = open('training_data.csv', 'w+')
training_data_csv.write('vector,label\n')

print(vectorized_training_data.shape[0])

for i, datum in enumerate(vectorized_training_data):
    training_data_csv.write('%s,%s\n' % (list(datum.toarray()[0]), label_list[i]))

training_data_csv.close()
