import pandas as pd
import json, collections
train = pd.read_csv("raw_training_data.csv", usecols=[0])


def build_wordlist(post):
    post = post.strip()
    post = post.split(' ')
    post = [x.strip() for x in post if x]
    return post

def build_dictionary(words):
    count = collections.Counter(words).most_common(10000)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = (len(dictionary) + 1)/10000
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

big_string = ""
for index, row in train.iterrows():
    big_string += str(row['post'])

wordlist = build_wordlist(big_string)
dictionary, reverse_dictionary = build_dictionary(wordlist)
vocab_size = len(dictionary)
print(dictionary)

with open("dictionary.json", 'w') as file:
    json.dump(dictionary, file)


print("DICTIONARY BUILT")