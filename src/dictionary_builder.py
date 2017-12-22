"""
Dictionary builder, using all the Reddit data.

Notes:
    - We take only the first n_words most common words to build our dictionary.
    - We also built a reverse dictionary, but it's never used.
    - All data is normalized between 0 and 1, for reasons of network structure
        So for example:
        - dict = {"hey": 1, "I": 2, "am": 3, "cool": 4}
            BECOMES
        - dict = {"hey": 0.25, "I": 0.50, "am": 0.75, "cool": 1.00}

By Jeff, Zac, Peter
"""


import pandas as pd
import json, collections
train = pd.read_csv("raw_training_data.csv", usecols=[0])


def build_wordlist(post):
    post = post.strip()
    post = post.split(' ')
    post = [x.strip() for x in post if x]
    return post

def build_dictionary(word_list, n_words):
    """ We take the first n_words most common words, otherwise our dictionary
        would be HUGE.
    """
    count = collections.Counter(word_list).most_common(n_words)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = (len(dictionary) + 1)/n_words
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
