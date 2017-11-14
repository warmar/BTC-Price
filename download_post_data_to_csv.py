import re
import os
import praw
import pickle
from pprint import pprint

post_files = os.listdir('post_data/')

done_post_ids = set()
with open('post_data.csv', 'r') as post_csv:
    raw_post_csv = post_csv.read()
    post_lines = raw_post_csv.split('\n')[1:-1]
    for post_line in post_lines:
        post_id = post_line.split(',')[0]
        done_post_ids.add(post_id)

post_csv = open('post_data.csv', 'a')

WORDS = re.compile(r'[a-zA-Z0-9\-]+')

done = 0
total = len(post_files)

for post_file in post_files:
    modified = False

    post_id = post_file
    if post_id not in done_post_ids:
        with open('post_data/%s' % post_file, 'rb') as pickle_file:
            # id,date,text,score,numcomments
            post = pickle.load(pickle_file)
            
            body = post.title + ' ' + post.selftext
            body = ' '.join(re.findall(WORDS, body))

            line = ''
            line += post_id + ','
            line += str(post.created_utc) + ','
            line += body + ','
            line += str(post.score) + ','
            line += str(len(post.comments))
            line += '\n'

            post_csv.write(line)

            modified = True

    done += 1

    if done % 10 == 0:
        print('%s of %s' % (done, total))
        if modified:
            post_csv.flush()

print('%s of %s' % (done, total))
post_csv.close()
