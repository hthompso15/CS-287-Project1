# Jarvis EDA
# for data white paper


######## Imports ##########################################################

import pandas as pd
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from scipy.stats import ttest_ind, ks_2samp


######## A. Globals #######################################################
cat =['GREET', 'JOKE', 'PIZZA', 'TIME', 'WEATHER']
col = ['character_length','word_count','characters_per_word', 'unique_word_count', 'stopword_count',
       'unique_vs_words', 'stopwords_vs_words', 'question_mark', 'period', 'compound', 'neg','neu']

################ B. Functions #############################################

def getCol(col, df, lst):
    '''
    get data from the dataframe using column name
    '''
    rt_lst = []
    for k in range(len(lst)):
      rt_lst.append(df[df['action'] == lst[k]][col])
    return rt_lst

def getCol2(col, df):
    val = df[col]
    return val

def count_unique_words(text):
    return len(set(text.split()))

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

# load data from database
#database_path = 'jarvis.db'
#connection = sqlite3.connect(database_path, check_same_thread=False)
#df = pd.read_sql("SELECT * FROM training_data", connection)
#df.to_csv('jarvis.csv', index=False)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# load data from csv
file = 'jarvis.csv'
df = pd.read_csv(file)
print(df.sample(5))

# check NaN
df.isna().sum()

# Simple Features from Raw Text
analyzer = SentimentIntensityAnalyzer()
df['character_length'] = df['message'].str.len()
df['word_count'] = df['message'].str.split().str.len()
df['characters_per_word'] = df['character_length']/df['word_count']
df['unique_word_count'] = df["message"].apply(lambda x:count_unique_words(x))
df['stopword_count'] = df["message"].apply(lambda x:count_stopwords(x))
df['unique_vs_words'] = df['unique_word_count']/df['word_count']
df['stopwords_vs_words'] = df['stopword_count']/df['word_count']
df['question_mark'] = df['message'].apply(lambda x: len([x for x in x.split() if x.endswith('?')]))
df['period'] = df['message'].apply(lambda x: len([x for x in x.split() if x.endswith('.')]))
df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['message']]
df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['message']]
df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['message']]
df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['message']]

# split the file into internal and external data
internal = df.iloc[0:64]
external = df.iloc[64:]

print(internal.head())
print(external.head())

# calcuate the action percentages of internal and external data
s1 = internal.groupby('action').message.count()
s2 = external.groupby('action').message.count()
s = pd.concat([s1, s2], axis=1).reset_index()
s.columns = ['action','internal','external']
s['internal'] = s['internal']/s['internal'].sum()
s['external'] = s['external']/s['external'].sum()

# make the clustered bar graph
ax = s.plot.bar(x='action', rot=0, figsize=(5,4))
ax.set_ylim(0.15,0.25)
ax.yaxis.grid(True)
ax.set_ylabel(r"%")
plt.savefig('P1.1.pdf')
plt.savefig('P1.1.png')
#plt.show()
plt.close()

# t-test to compare external and internal data
for i in range(len(col)):
    print(col[i])
    print(ttest_ind(getCol2(col[i], external), getCol2(col[i], internal), axis=0, equal_var=True))


internal_stats = internal.describe().loc[['mean', 'std', 'min', '50%', 'max']].T
internal_stats.insert (0, 'type', 'internal')
external_stats = external.describe().loc[['mean', 'std', 'min', '50%', 'max']].T
with open("external_stats.txt", "a") as f:
    print(external_stats.sort_index(ascending=True),file=f)
external_stats.insert (0, 'type', 'external')


all_stats = pd.concat([internal_stats, external_stats])
with open("all_stats.txt", "a") as f:
    print(all_stats.sort_index(ascending=True),file=f)


fig, axes = plt.subplots(4, 3, figsize=(8, 10.5), dpi=100, sharex=False, sharey=True)
axes = axes.ravel()

#remove compound to save space
col = ['character_length','word_count','characters_per_word', 'unique_word_count', 'stopword_count',
       'unique_vs_words', 'stopwords_vs_words', 'question_mark', 'period', 'neg','neu', 'pos']

for k, ax in enumerate(axes):
    for j in range(len(cat)):
        x = sorted(getCol(col[k], external, cat)[j])
        n = len(x);
        y = [i/n for i in range(n)]
        ax.plot(x,y,'.-', alpha = 0.4, label=cat[j])
    #ax.legend(loc='upper left')
    ax.set_title(col[k])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels,  loc = (-1.75, -0.35), ncol=5)
fig.text(-0.05, 0.5, r'p(x)', va='center', rotation='vertical')


fig.savefig('P1.2.pdf')
fig.savefig('P1.2.svg')
plt.show()
plt.close()

## K-S test
for i in range(len(cat)):
  for j in range(len(cat)):
      print(cat[i], cat[j])
      print(ks_2samp((getCol('pos', external, cat)[i]), (getCol('pos', external, cat)[j])))

pizza = df.groupby("action").get_group("PIZZA")["message"].tolist()
weather = df.groupby("action").get_group("WEATHER")["message"].tolist()
greet = df.groupby("action").get_group("GREET")["message"].tolist()
time = df.groupby("action").get_group("TIME")["message"].tolist()
joke = df.groupby("action").get_group("JOKE")["message"].tolist()

lst_lst = [greet, joke, pizza, time, weather]
lst_name = ['greet', 'joke', 'pizza', 'time', 'weather']

tfidf = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 2))

fig, axes = plt.subplots(5, 1, figsize=(8, 10.5), dpi=200, sharex=False, sharey=True)
axes = axes.ravel()

for k, ax in enumerate(axes):
    matrix = tfidf.fit_transform(lst_lst[k])
    feature_names = tfidf.get_feature_names()
    dense_matrix = matrix.todense()
    lst = dense_matrix.tolist()
    df2 = pd.DataFrame(lst, columns=feature_names)
    df2 = df2.T.sum(axis=1)
    wc = WordCloud(background_color="white", max_words=100, width=800,height=200).generate_from_frequencies(df2)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(lst_name[k])
    ax.axis("off")

fig.savefig('P1.3.pdf')
fig.savefig('P1.3.svg')
fig.savefig('P1.3.png')
plt.show()
plt.close()