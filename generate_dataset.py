#!/usr/bin/env python
# coding: utf-8

import gzip
import json
import pandas as pd
import nltk
from os import listdir
from nltk.tokenize import word_tokenize
from argparse import ArgumentParser
from tqdm import tqdm


def get_arguments():
    """Get arguments from the user

    Returns:
        [ArgumentParser Object] -- Object containing all the arguments
    """
    opt = ArgumentParser()

    opt.add_argument(
        "--max_tweets", default=None, help="unlimited or integer to limit the number of tweets",
    )
    opt.add_argument("--tweet_type", default="generic", help="generic or toxic")

    return opt.parse_args()


def collect_tweets(tweet_type, max_tweets):
    print("Collecting tweets...")
    u_ids = []
    u_names = []
    u_locations = []
    friends_count = []
    u_descriptions = []
    verifications = []
    followers_count = []
    u_tweets_count = []
    u_favorites_count = []
    years_creation = []

    tweet_ids = set()
    texts = set()
    urls = []
    mentions = []
    hashtags = []
    tweets_replies = []
    tweets_favorites = []
    data = pd.DataFrame(
        columns=[
            "user_id",
            "user_name",
            "user_location",
            "user_description",
            "user_friends_count",
            "user_is_verified",
            "user_followers_count",
            "user_tweets_count",
            "user_favorites_count",
            "user_created_at",
            "tweet_text",
            "tweet_urls",
            "tweet_mentions",
            "tweet_hashtags",
            "tweet_replies",
            "tweet_favorites",
        ]
    )

    PATH = f"data/raw_data/{tweet_type}/"
    print([PATH + f for f in listdir(PATH) if (f.endswith(".gz"))][:2])
    file_list = [PATH + f for f in listdir(PATH) if (f.endswith(".gz"))][:2]
    print("# of files:", len(file_list))
    for i, file in tqdm(enumerate(file_list)):
        with gzip.open(file) as f:
            file_lines = f.readlines()
            for line in file_lines:
                parsed_line = json.loads(line)
                try:  # Retweet
                    # text
                    tweet_text = parsed_line["retweeted_status"]["extended_tweet"]["full_text"]

                    # urls
                    tweet_urls = []
                    parsed_urls = parsed_line["retweeted_status"]["extended_tweet"]["entities"][
                        "urls"
                    ]
                    for parsed_url in parsed_urls:
                        if parsed_url:  # if list of urls is not empty
                            tweet_urls.append(parsed_url["url"])

                    # mentions
                    tweet_mentions = []
                    parsed_mentions = parsed_line["retweeted_status"]["extended_tweet"]["entities"][
                        "user_mentions"
                    ]
                    for parsed_mention in parsed_mentions:
                        if parsed_mention:
                            tweet_mentions.append(parsed_mention["screen_name"])

                    # hashtags
                    tweet_hashtags = []
                    parsed_hashtags = parsed_line["retweeted_status"]["extended_tweet"]["entities"][
                        "hashtags"
                    ]
                    for parsed_hashtag in parsed_hashtags:
                        if parsed_hashtag:
                            tweet_hashtags.append(parsed_hashtag["text"])

                    # reply counts
                    tweet_replies = parsed_line["retweeted_status"]["reply_count"]

                    # favorite counts
                    tweet_favorites = parsed_line["retweeted_status"]["favorite_count"]
                except KeyError:  # Long Tweet
                    try:
                        # text
                        tweet_text = parsed_line["extended_tweet"]["full_text"]

                        # urls
                        tweet_urls = []
                        parsed_urls = parsed_line["extended_tweet"]["entities"]["urls"]
                        for parsed_url in parsed_urls:
                            if parsed_url:  # if list of urls is not empty
                                tweet_urls.append(parsed_url["url"])

                        # mentions
                        tweet_mentions = []
                        parsed_mentions = parsed_line["extended_tweet"]["entities"]["user_mentions"]
                        for parsed_mention in parsed_mentions:
                            if parsed_mention:
                                tweet_mentions.append(parsed_mention["screen_name"])

                        # hashtags
                        tweet_hashtags = []
                        parsed_hashtags = parsed_line["extended_tweet"]["entities"]["hashtags"]
                        for parsed_hashtag in parsed_hashtags:
                            if parsed_hashtag:
                                tweet_hashtags.append(parsed_hashtag["text"])

                        # reply counts
                        tweet_replies = parsed_line["reply_count"]

                        # favorite counts
                        tweet_favorites = parsed_line["favorite_count"]
                    except KeyError:  # Short Tweet
                        # text
                        tweet_text = parsed_line["text"].lower()

                        # urls
                        tweet_urls = []
                        parsed_urls = parsed_line["entities"]["urls"]
                        for parsed_url in parsed_urls:
                            if parsed_url:  # if list of urls is not empty
                                tweet_urls.append(parsed_url["url"])

                        # mentions
                        tweet_mentions = []
                        parsed_mentions = parsed_line["entities"]["user_mentions"]
                        for parsed_mention in parsed_mentions:
                            if parsed_mention:
                                tweet_mentions.append(parsed_mention["screen_name"])

                        # hashtags
                        tweet_hashtags = []
                        parsed_hashtags = parsed_line["entities"]["hashtags"]
                        for parsed_hashtag in parsed_hashtags:
                            if parsed_hashtag:
                                tweet_hashtags.append(parsed_hashtag["text"])

                        # reply counts
                        tweet_replies = parsed_line["reply_count"]

                        # favorite counts
                        tweet_favorites = parsed_line["favorite_count"]
                finally:  # User information
                    # tweet id
                    tweet_id = parsed_line["id"]

                    # user id
                    user_id = parsed_line["user"]["id"]

                    # user name
                    user_name = parsed_line["user"]["screen_name"]

                    # user location
                    user_location = parsed_line["user"]["location"]

                    # friends count
                    user_friends_count = parsed_line["user"]["friends_count"]

                    # user description
                    user_description = parsed_line["user"]["description"]

                    # verified
                    verified = parsed_line["user"]["verified"]

                    # followers count
                    user_followers_count = parsed_line["user"]["followers_count"]

                    # tweets count
                    user_tweets_count = parsed_line["user"]["statuses_count"]

                    # favorites count
                    user_favorites_count = parsed_line["user"]["favourites_count"]

                    # year of creation
                    user_creation_year = parsed_line["user"]["created_at"][-4:]

                # append unique tweets
                if tweet_text not in texts:
                    texts.add(tweet_text)
                    tweet_ids.add(tweet_id)
                    u_ids.append(user_id)
                    u_names.append(user_name)
                    u_locations.append(user_location)
                    friends_count.append(user_friends_count)
                    u_descriptions.append(user_description)
                    verifications.append(verified)
                    followers_count.append(user_followers_count)
                    u_tweets_count.append(user_tweets_count)
                    u_favorites_count.append(user_favorites_count)
                    years_creation.append(user_creation_year)

                    urls.append(tweet_urls)
                    mentions.append(tweet_mentions)
                    hashtags.append(tweet_hashtags)
                    tweets_replies.append(tweet_replies)
                    tweets_favorites.append(tweet_favorites)

                    if max_tweets and len(texts) >= max_tweets:
                        break

    print("total amount of tweets:", len(texts))

    data["user_id"] = u_ids
    del u_ids
    data["user_name"] = u_names
    del u_names
    data["user_location"] = u_locations
    del u_locations
    data["user_friends_count"] = friends_count
    del friends_count
    data["user_description"] = u_descriptions
    del u_descriptions
    data["user_is_verified"] = verifications
    del verifications
    data["user_followers_count"] = followers_count
    del followers_count
    data["user_tweets_count"] = u_tweets_count
    del u_tweets_count
    data["user_favorites_count"] = u_favorites_count
    del u_favorites_count
    data["user_created_at"] = years_creation
    del years_creation

    data["tweet_text"] = texts
    del texts
    data["tweet_urls"] = urls
    del urls
    data["tweet_mentions"] = mentions
    del mentions
    data["tweet_hashtags"] = hashtags
    del hashtags
    data["tweet_replies"] = tweets_replies
    del tweets_replies
    data["tweet_favorites"] = tweets_favorites
    del tweets_favorites

    return data


if __name__ == "__main__":
    args = get_arguments()

    data = collect_tweets(
        tweet_type=args.tweet_type, max_tweets=int(args.max_tweets) if args.max_tweets else 0
    )
    data.to_csv(f"data/raw_data/dataset/dataset_{args.tweet_type}.csv", header=True, index=False)
