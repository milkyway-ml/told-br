"""Generate word embeddings from raw tweets."""
import gzip
import json
from numpy import inf
from os import listdir
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from argparse import ArgumentParser

RAW_TWEETS_PATH = "data/raw_data/"


def get_arguments():
    """Get arguments from the user

    Returns:
        [ArgumentParser Object] -- Object containing all the arguments
    """
    opt = ArgumentParser()

    opt.add_argument(
        "--max_tweets",
        default="unlimited",
        help="unlimited or integer to limit the number of tweets",
    )
    opt.add_argument("--tweet_type", default="generic", help="generic or toxic")
    opt.add_argument("--window_size", default=5, help="gensim window size for embeddings")
    opt.add_argument("--vector_dim", default=300, help="gensim vector dimension of embeddings")

    return opt.parse_args()


def get_tweets(path, max_tweets):
    """Extract tweets from compressed json files.

    Arguments:
        path {str} -- path to the directory containing .json.gz files with raw tweets

    Keyword Arguments:
        max_tweets {int} -- numpy.inf for unlimited tweets,
                            integer to limit the amout of extract tweets

    Returns:
        [list of str] -- list with all the tweets extracted
    """
    file_list = [path + f for f in listdir(path)[2:] if (f.endswith(".gz"))]

    # configure tqdm according to max_tweets
    if max_tweets != inf:
        flag_max_tweets = True
        total = max_tweets
    else:
        flag_max_tweets = False
        total = len(file_list)

    print("Extracting text from tweets...")
    exit_flag = False
    data = set()
    with tqdm(total=total) as progressbar:
        for i, file_name in enumerate(file_list):
            if exit_flag:
                break
            else:
                with gzip.open(file_name) as f:
                    try:
                        file_lines = f.readlines()
                    except EOFError:
                        continue
                    else:
                        for j, line in enumerate(file_lines):
                            parsed_line = json.loads(line)
                            try:  # Retweet
                                tweet_text = parsed_line["retweeted_status"]["extended_tweet"][
                                    "full_text"
                                ]
                            except KeyError:  # Long Tweet
                                try:
                                    tweet_text = parsed_line["extended_tweet"]["full_text"]
                                except KeyError:  # Short Tweet
                                    tweet_text = parsed_line["text"].lower()

                            if flag_max_tweets and tweet_text not in data:
                                progressbar.update(1)

                            data.add(tweet_text)

                            if len(data) >= max_tweets:
                                exit_flag = True
                                break

            if not flag_max_tweets:
                progressbar.update(1)

    print("Total amount of unique tweets: {}".format(len(data)))
    return list(data)


def generate_bigrams(data, min_count=20, delimiter=b"_"):
    """Connect two tokens that appear togheter many times.

    Arguments:
        data {iterable} -- iterable containing all the tweets

    Keyword Arguments:
        min_count {int} -- minimum amount of apearences two tokens
                            must appear togheter to form a new single token (default: {20})
        delimiter {bytes} -- byte char that  will connect two tokens (default: {b" "})

    Returns:
        [list] -- list of str contaning tweets with tokens connected by '_'
    """
    print("Generating bigrams...")
    phrases = Phrases(data, min_count=min_count, delimiter=delimiter)
    bigram = Phraser(phrases)
    sentences = bigram[data]

    return sentences


def generate_embeddings(sentences, num_tweets, tweet_type, size=300, window=5, sg=0):
    """Construct continuous bag of words and skipgram from input text.

    Arguments:
        sentences {list of str} -- input text
        num_tweets {int} -- total amount of tweets (used only to label the final file)
        tweet_type {str} -- toxic or generic (only used to label the final file)

    Keyword Arguments:
        size {int} -- dimension of each word embedding. (default: {300})
        window {int} -- size of the window that considers the proximity of the tokens.(default: {5})
        sg {int} -- 0 for continuous bag of words, 1 for skipgram. (default: {0})
    """
    if sg == 0:
        embedding_type = "cbow"
    elif sg == 1:
        embedding_type = "skipgram"

    print(
        "Generating {} with dimension={} and window={}".format(
            embedding_type, str(size), str(window)
        )
    )

    embeddings = Word2Vec(sentences, size=size, window=window, sg=sg)
    embeddings.save(
        "data/word_embeddings/twitter_{}_{}_{}_{}_{}".format(
            tweet_type, embedding_type, size, window, num_tweets
        )
    )


if __name__ == "__main__":
    opt = get_arguments()
    path = RAW_TWEETS_PATH + opt.tweet_type + "/"

    max_tweets = opt.max_tweets
    if max_tweets == "unlimited":
        max_tweets = inf
    else:
        max_tweets = int(max_tweets)

    window_size = int(opt.window_size)
    vector_dim = int(opt.vector_dim)

    data = get_tweets(path=path, max_tweets=max_tweets)
    sentences = generate_bigrams(data=data)
    generate_embeddings(
        sentences=sentences,
        tweet_type=opt.tweet_type,
        num_tweets=len(data),
        size=vector_dim,
        window=window_size,
        sg=0,
    )
    generate_embeddings(
        sentences=sentences,
        tweet_type=opt.tweet_type,
        num_tweets=len(data),
        size=vector_dim,
        window=window_size,
        sg=1,
    )
