import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
# remove morphological affixes from words, leaving only the word stem
english_stemmer = nltk.stem.SnowballStemmer('english')
from sklearn.feature_extraction.text import CountVectorizer


# Reading the data provided via http://jmcauley.ucsd.edu/data/amazon/
def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Removing stop wgtords with NLTK
def data_clean(rev, remove_stopwords=True):
    new_text = re.sub("[^a-zA-Z]", " ", rev)

    words = new_text.lower().split()

    if remove_stopwords:
        sts = set(stopwords.words("english")) # filter out useless data
        words = [w for w in words if not w in sts]
    ary = []
    eng_stemmer = english_stemmer
    for word in words:
        ary.append(eng_stemmer.stem(word))

    return (ary)

def vectorizer(corpus):
    vect = CountVectorizer()
    X = vect.fit_transform(corpus)
    X_feature = vect.get_feature_names()

    return X, X_feature
