import time
import os
import json
import praw

# Declare Parameters
REDDIT_CLIENT_ID = '-oaj412g-c_iVg'
REDDIT_CLIENT_SECRET = 'Qp7Vj5MHw42TeupmIPR_kwZpd-s'
REDDIT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'

SUBREDIT_NAME = 'bitcoin'
START_TIMESTAMP = 1313331280
END_TIMESTAMP = 1500330893

print(time.ctime(START_TIMESTAMP), ' to ', time.ctime(END_TIMESTAMP))

# Initialize reddit instances
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
btc = reddit.subreddit(SUBREDIT_NAME)

# Determine earliest already-downloaded post timestamp (posts download from most recent to oldest)
done = 0

earliest_timestamp = time.time()
post_file_names = os.listdir('post_data')
for post_file_name in post_file_names:
    with open('post_data/%s' % post_file_name, 'r') as post_file:
        post = json.loads(post_file.read())
    
    earliest_timestamp = min(earliest_timestamp, post['timestamp'])

    done += 1

# Download remaining posts
for submission in btc.submissions(START_TIMESTAMP, min(earliest_timestamp, END_TIMESTAMP)):
    if submission.is_self:
        submission_json = {
            'id': submission.id,
            'timestamp': submission.created_utc,
            'score': submission.score,
            'numcomments': len(submission.comments),
            'title': submission.title,
            'selftext': submission.selftext
        }

        # Write submission json objects to separate files in post_data/
        with open('post_data/%s' % submission.id, 'w+') as post_data_file:
            post_data_file.write(json.dumps(submission_json))

    done += 1
    if done % 100 == 0:
        print('%s done - %s' % (done, time.ctime(submission_json['timestamp'])))

print('%s done - %s' % (done, time.ctime(submission_json['timestamp'])))
